#include "TripletTranslationsValidator.h"
#include "Triangulator.h"
#include "CeresBA.h"

TripletTranslationsValidator::TripletTranslationsValidator(Options options)
	: m_options(options)
{
}

bool TripletTranslationsValidator::acceptable(
	RelativeTransform & transform_ij, 
	RelativeTransform & transform_ik, 
	std::shared_ptr<Camera> camera_i,
	std::shared_ptr<Camera> camera_j,
	std::shared_ptr<Camera> camera_k,
	Tracks & tracks
)
{
	GlobalTransform Pi;
	Pi.is_inlier = true;
	Pi.view_id = transform_ij.src_id;
	Pi.rt = CameraExtrinsic::Zero();

	GlobalTransform Pj;
	Pj.is_inlier = true;
	Pj.view_id = transform_ij.dst_id;
	Pj.rt = transform_ij.rt;

	GlobalTransform Pk;
	Pk.is_inlier = true;
	Pk.view_id = transform_ik.dst_id;
	Pk.rt = transform_ik.rt;

	Scene tiny_scene;
	tiny_scene.tracks.swap(tracks);
	tiny_scene.views.push_back(Pi);
	tiny_scene.views.push_back(Pj);
	tiny_scene.views.push_back(Pk);
	tiny_scene.view_idx_by_view_id[Pi.view_id] = 0;
	tiny_scene.view_idx_by_view_id[Pj.view_id] = 1;
	tiny_scene.view_idx_by_view_id[Pk.view_id] = 2;
	tiny_scene.view_camera_binder.bind(Pi.view_id, camera_i);
	tiny_scene.view_camera_binder.bind(Pj.view_id, camera_j);
	tiny_scene.view_camera_binder.bind(Pk.view_id, camera_k);

	bool accepted = acceptable(tiny_scene);
	if (accepted)
	{
		// The first view's extrinsic is fixed in BA progress. 
		// So the global transforms of the second and third view are equal to the relation transforms respectively.
		transform_ij.rt = tiny_scene.views[1].rt;
		transform_ik.rt = tiny_scene.views[2].rt;
	}

	tracks.swap(tiny_scene.tracks);
	return accepted;
}

bool TripletTranslationsValidator::acceptable(Scene & three_views_scene)
{
	Triangulator triangulator;

	size_t num_inliers = triangulator.multi_views_dlt(three_views_scene, false, m_options.init_max_reproj_error, 3, 3);
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

		// adjust translations and structures only.
		if (ba.solve(three_views_scene, CeresBA::ADJUST_TRANSLATIONS | CeresBA::ADJUST_STRUCTURE))
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
