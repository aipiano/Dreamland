#include "RefineRelativeTranslationsWithKnownRotations.h"
#include "Ransac.hpp"
#include "RansacKernel.hpp"
#include "Utils.h"
#include <opencv2\core.hpp>

using namespace Eigen;
using namespace std;

typedef Matrix<double, 6, 1> PointPairVector;

class RefineTranslationKernel final : public RansacKernel<PointPairVector, Vector3d>
{
public:
	RefineTranslationKernel(const Eigen::Ref<const Eigen::Matrix3d>& R)
	{
		set_rotation(R);
	}

	// Inherited via RansacKernel
	virtual void run_kernel(const std::vector<PointPairVector>& samples, std::vector<Vector3d>& models) override
	{
		assert(samples.size() >= 3);

		MatrixXd A(samples.size(), 3);
		for (size_t i = 0; i < samples.size(); ++i)
		{
			Vector3d x1 = samples[i].topRows(3);
			Vector3d x2 = samples[i].bottomRows(3);

			A.row(i).noalias() = ((m_R*x1).cross(x2)).transpose();
		}

		Vector3d T = A.jacobiSvd(ComputeFullV).matrixV().rightCols(1);

		models.resize(1);
		models[0] = T;
	}

	virtual void compute_error(const std::vector<PointPairVector>& samples, const Vector3d & model, Eigen::VectorXd & errors) override
	{
		for (size_t i = 0; i < samples.size(); ++i)
		{
			Vector3d x1 = samples[i].topRows(3);
			Vector3d x2 = samples[i].bottomRows(3);

			errors[i] = x2.dot(model.cross(m_R * x1));
		}
	}

	void set_rotation(const Eigen::Ref<const Eigen::Matrix3d>& R)
	{
		m_R = R;
	}

private:
	Matrix3d m_R;

};

void refine_relative_translations_with_known_rotations(
	EpipolarGraph & epi_graph, 
	std::shared_ptr<FeaturePool> feature_pool, 
	std::shared_ptr<MatchPool> match_pool,
	ViewCameraBinder& view_camera_binder
)
{
	EpipolarGraph::edge_iterator e_begin, e_end;
	tie(e_begin, e_end) = edges(epi_graph);

	// for openmp parallel usaging.
	vector<EpipolarGraph::edge_descriptor> all_edges;
	all_edges.reserve(num_edges(epi_graph));
	for (auto it_e = e_begin; it_e != e_end; ++it_e)
	{
		all_edges.push_back(*it_e);
	}
	int edge_count = all_edges.size();

	vector<PointPairVector> samples;
	vector<PointPairVector> inliers;
	vector<uchar> mask;

	// TODO: parallel
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

		//获取train_id和query_id对应的相机
		auto train_camera = view_camera_binder.get_camera(train_id);
		auto query_camera = view_camera_binder.get_camera(query_id);

		auto num_matches = matches.size();
		samples.resize(num_matches);
		
		auto x1_idx = (id1 == train_id ? 0 : 1);
		auto x2_idx = 1 - x1_idx;

		// prepare ransac samples
		for (size_t i = 0; i < num_matches; ++i)
		{
			auto& pt1 = train_kps[matches[i].trainIdx].pt;
			auto& pt2 = query_kps[matches[i].queryIdx].pt;

			auto world_pt1 = train_camera->image_to_world(Vector2d(pt1.x, pt1.y));
			auto world_pt2 = query_camera->image_to_world(Vector2d(pt2.x, pt2.y));

			samples[i].middleRows(x1_idx * 3, 3) = world_pt1;
			samples[i].middleRows(x2_idx * 3, 3) = world_pt2;
		}

		Matrix3d R;
		angle_axis_to_rotation_matrix(epi_graph[e].rt.topRows(3), R);
		auto refine_kernel = make_shared<RefineTranslationKernel>(R);

		double focal = train_camera->get_intrinsic()[0] + query_camera->get_intrinsic()[0];	//TODO: just for test
		focal /= 2.0;
		Ransac<PointPairVector, Vector3d> ransac(refine_kernel, 3, 3. / focal);

		// do refinement
		Vector3d model;
		if (!ransac.run(samples, model, mask))
			continue;

		auto num_inliers = cv::countNonZero(mask);
		if (num_inliers < 30)	// TODO: make the minimum inliers' count a parameter
			continue;

		inliers.resize(num_inliers);
		size_t inlier_idx = 0;
		for (size_t i = 0; i < mask.size(); ++i)
		{
			if (mask[i] == 0) continue;
			inliers[inlier_idx] = samples[i];
			++inlier_idx;
		}

		// estimate translation again by all inliers.
		vector<Vector3d> models;
		refine_kernel->run_kernel(inliers, models);
		model = models[0];

		// update translation
		if (epi_graph[e].rt.bottomRows(3).dot(model) < 0)
			model *= -1;

		//auto& ori_transform = epi_graph[e].rt;
		epi_graph[e].rt.bottomRows(3) = model;
	}
}
