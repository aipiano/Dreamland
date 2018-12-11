#pragma once

#include "Scene.h"
#include "Camera.h"

class TripletTranslationsValidator
{
public:
	struct Options
	{
		size_t min_num_inliers = 30;			// min inliers satisfy the init_max_reproj_error
		double init_max_reproj_error = 5.0;		// max allowed reproject error after triangulate
		double final_max_reproj_error = 2.0;	// max allowed reproject error after BA
		bool refine_with_ba = true;				// refine translations with BA
	};

	TripletTranslationsValidator(Options options);

	bool acceptable(
		RelativeTransform& transform_ij,
		RelativeTransform& transform_ik,
		std::shared_ptr<Camera> camera_i,
		std::shared_ptr<Camera> camera_j,
		std::shared_ptr<Camera> camera_k,
		Tracks& tracks
	);

	bool acceptable(Scene& three_views_scene);

private:
	Options m_options;
};