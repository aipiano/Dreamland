#include "PairwiseTransformValidator.h"
#include "GlobalTransform.h"
#include "Scene.h"
#include "Triangulator.h"
#include "CeresBA.h"
#include "Utils.h"

using namespace Eigen;
using namespace std;

PairwiseTransformValidator::PairwiseTransformValidator(Options & options)
	: m_options(options)
{
}

bool PairwiseTransformValidator::acceptable(
	RelativeTransform & transform, 
	std::shared_ptr<Camera> src_camera,
	std::shared_ptr<Camera> dst_camera,
	std::vector<Eigen::Vector2d>& src_observations,
	std::vector<Eigen::Vector2d>& dst_observations
)
{
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
	tiny_scene.view_camera_binder.bind(transform.src_id, src_camera);
	tiny_scene.view_camera_binder.bind(transform.dst_id, dst_camera);

	for (size_t i = 0; i < src_observations.size(); ++i)
	{
		Vector2d& observe_pt1 = src_observations[i];
		Vector2d& observe_pt2 = dst_observations[i];

		Track track;
		track.is_inlier = true;
		TrackNode node1(observe_pt1, transform.src_id);
		TrackNode node2(observe_pt2, transform.dst_id);
		track.nodes.push_back(node1);
		track.nodes.push_back(node2);

		tiny_scene.tracks.push_back(std::move(track));
	}

	bool accepted = acceptable(tiny_scene);
	if (accepted)
	{
		//auto& refined_extrinsic_src = tiny_scene.views[0].rt;	//P0
		//auto& refined_extrinsic_dst = tiny_scene.views[1].rt;	//P1
		//relative_transform_between(refined_extrinsic_src, refined_extrinsic_dst, transform.rt);

		// The first view's extrinsic is fixed in BA progress. 
		// So the global transform of the second view is equal to the relation transform.
		transform.rt = tiny_scene.views[1].rt;
	}

	return accepted;
}

bool PairwiseTransformValidator::acceptable(Scene & two_views_scene)
{
	Triangulator triangulator;

	size_t num_inliers = triangulator.multi_views_dlt(two_views_scene, false, m_options.init_max_reproj_error);
	if (num_inliers < m_options.min_num_inliers)
		return false;

	if (m_options.refine_with_ba)
	{
		CeresBA::Options options;
#ifdef _DEBUG
		options.show_verbose = true;
#else
		options.show_verbose = false;
#endif
		options.show_ceres_summary = false;
		options.multithreaded = false;
		CeresBA ba(options);

		// adjust extrinsics and structure only
		if (ba.solve(two_views_scene, CeresBA::ADJUST_EXTRINSICS | CeresBA::ADJUST_STRUCTURE))
		{
			if (ba.final_rmse() > m_options.final_max_reproj_error)
				return false;
		}
		else
		{
			return false;
		}
	}

	return true;
}
