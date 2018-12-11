#include "FivePointsEstimator.h"
#include "CeresBA.h"
#include "PairwiseTransformValidator.h"
#include "Utils.h"
#include <opencv2\calib3d.hpp>
#include <Eigen/Eigen>
#include <omp.h>

using namespace std;
using namespace boost;
using namespace cv;
using namespace Eigen;

FivePointsEstimator::FivePointsEstimator(Options & options)
	:m_options(options)
{
}

bool FivePointsEstimator::estimate(
	std::shared_ptr<FeaturePool> feature_pool, 
	std::shared_ptr<MatchPool> match_pool, 
	ViewCameraBinder& view_camera_binder,
	EpipolarGraph & epi_graph
)
{
	EpipolarGraph::edge_iterator e_begin, e_end;
	tie(e_begin, e_end) = edges(epi_graph);

	PairwiseTransformValidator::Options validator_options;
	validator_options.refine_with_ba = m_options.refine_with_ba;
	validator_options.min_num_inliers = m_options.min_num_inliers;
	validator_options.init_max_reproj_error = m_options.init_max_reproj_error;
	validator_options.final_max_reproj_error = m_options.final_max_reproj_error;
	PairwiseTransformValidator validator(validator_options);
	
	// for openmp parallel usaging.
	vector<EpipolarGraph::edge_descriptor> all_edges;
	all_edges.reserve(num_edges(epi_graph));
	for (auto it_e = e_begin; it_e != e_end; ++it_e)
	{
		all_edges.push_back(*it_e);
	}

	omp_set_num_threads(omp_get_num_procs());
	int edge_count = all_edges.size();

	vector<Point2d> points1, points2;
	Mat mask;

	#pragma omp parallel for schedule(dynamic) private(mask, points1, points2)
	for (int e_idx = 0; e_idx < edge_count; ++e_idx)
	{
		auto& e = all_edges[e_idx];
		auto id1 = epi_graph[e.m_source].view_id;
		auto id2 = epi_graph[e.m_target].view_id;
		
		auto match_frame = match_pool->get_matches(id1, id2);
		if (match_frame == nullptr)
		{
			epi_graph[e].is_inlier = false;
			continue;
		}
		//MatchPair中的ID和MatchFrame中的ID可能相反，所以使用MatchFrame中的ID获取特征点
		auto train_id = match_frame->train_id;
		auto query_id = match_frame->query_id;
		auto& matches = match_frame->matches;
		
		// hold pointers to add reference count
		auto train_kps_ptr = feature_pool->get_keypoints(train_id);
		auto query_kps_ptr = feature_pool->get_keypoints(query_id);
		const auto& train_kps = *train_kps_ptr;
		const auto& query_kps = *query_kps_ptr;

		auto num_matches = matches.size();
		points1.clear();
		points2.clear();
		points1.reserve(num_matches);
		points2.reserve(num_matches);
		
		//获取train_id和query_id对应的相机
		auto train_camera = view_camera_binder.get_camera(train_id);
		auto query_camera = view_camera_binder.get_camera(query_id);
		
		for (size_t i = 0; i < num_matches; ++i)
		{
			auto& pt1 = train_kps[matches[i].trainIdx].pt;
			auto& pt2 = query_kps[matches[i].queryIdx].pt;

			auto world_pt1 = train_camera->image_to_world(Vector2d(pt1.x, pt1.y));
			auto world_pt2 = query_camera->image_to_world(Vector2d(pt2.x, pt2.y));

			points1.push_back(Point2d(world_pt1[0] / world_pt1[2], world_pt1[1] / world_pt1[2]));
			points2.push_back(Point2d(world_pt2[0] / world_pt2[2], world_pt2[1] / world_pt2[2]));
		}

		//获取从train_id到query_id的本征矩阵(from points1 to points2)
		auto& train_intrinsic = train_camera->get_intrinsic();
		auto& query_intrinsic = query_camera->get_intrinsic();
		//TODO: 寻找其他不依赖内参的阈值
		double avg_focal = (train_intrinsic[0] + train_intrinsic[1] + query_intrinsic[0] + query_intrinsic[1]) / 4.0;

		//TODO: 自己实现findEssentialMat方法
		Mat E = findEssentialMat(points1, points2, 1.0, Point2d(0, 0), RANSAC, 0.999, m_options.ransac_threshold / avg_focal, mask);

		int num_inliers = countNonZero(mask);
		if (E.empty() || num_inliers < m_options.min_num_inliers)	//find essential matrix failed
		{
			epi_graph[e].is_inlier = false;
			continue;
		}

		Matx33d R;
		Matx31d t;
		//TODO: 自己实现recoverPose，需要返回反向投影误差
		int pass_count = recoverPose(E, points1, points2, R, t, 1.0, Point2d(0, 0), mask);
		if (pass_count < m_options.min_num_inliers)	// too few point in front of camera
		{
			epi_graph[e].is_inlier = false;
			continue;
		}

		Matx31d r;
		Rodrigues(R, r);
		RelativeTransform& transform = epi_graph[e];
		transform.is_inlier = true;
		transform.src_id = train_id;
		transform.dst_id = query_id;
		transform.weight = double(num_inliers) / double(points1.size());
		copy2extrinsic(r, t, transform.rt);

		// x2 = R*x1 + T
		GlobalTransform P0;
		P0.is_inlier = true;
		P0.view_id = transform.src_id;
		P0.rt = CameraExtrinsic::Zero();

		GlobalTransform P1;
		P1.is_inlier = true;
		P1.view_id = transform.dst_id;
		P1.rt = transform.rt;

		Scene tiny_scene;
		tiny_scene.views.push_back(P0);	// idx = 0
		tiny_scene.views.push_back(P1);	// idx = 1
		tiny_scene.view_idx_by_view_id[P0.view_id] = 0;
		tiny_scene.view_idx_by_view_id[P1.view_id] = 1;
		tiny_scene.view_camera_binder.bind(transform.src_id, train_camera);
		tiny_scene.view_camera_binder.bind(transform.dst_id, query_camera);

		unsigned char* ptr_mask = mask.ptr<unsigned char>();
		for (size_t i = 0; i < matches.size(); ++i)
		{
			if (ptr_mask[i] == 0) continue;

			auto& src_observe_pt = train_kps[matches[i].trainIdx].pt;
			auto& dst_observe_pt = query_kps[matches[i].queryIdx].pt;

			Track track;
			track.is_inlier = true;
			TrackNode node1(Vector2d(src_observe_pt.x, src_observe_pt.y), transform.src_id);
			TrackNode node2(Vector2d(dst_observe_pt.x, dst_observe_pt.y), transform.dst_id);
			track.nodes.push_back(node1);
			track.nodes.push_back(node2);

			tiny_scene.tracks.push_back(std::move(track));
		}

		if (validator.acceptable(tiny_scene))
		{
			//auto& refined_extrinsic_src = tiny_scene.views[0].rt;	//P0
			//auto& refined_extrinsic_dst = tiny_scene.views[1].rt;	//P1
			//relative_transform_between(refined_extrinsic_src, refined_extrinsic_dst, transform.rt);

			// The first view's extrinsic is fixed in BA progress. 
			// So the global transform of the second view is equal to the relation transform.
			transform.rt = tiny_scene.views[1].rt;
			#pragma omp critical
			{
				cout << "Pairwise Transform " << id1 << " - " << id2 << ": (#inlier / #total): " << num_inliers << " / " << points1.size() << endl;
			}
		}
		else
		{
			epi_graph[e].is_inlier = false;
		}
	}

	return true;
}

inline void FivePointsEstimator::copy2extrinsic(cv::Matx31d & r, cv::Matx31d & t, CameraExtrinsic & rt)
{
	rt[0] = r(0);
	rt[1] = r(1);
	rt[2] = r(2);

	rt[3] = t(0);
	rt[4] = t(1);
	rt[5] = t(2);
}

