#pragma once

#include "RelativeTransform.h"
#include "Camera.h"
#include "Scene.h"
#include <memory>
#include <vector>

class PairwiseTransformValidator
{
public:
	struct Options
	{
		size_t min_num_inliers = 30;			// min inliers satisfy the init_max_reproj_error
		double init_max_reproj_error = 5.0;		// max allowed reproject error after triangulate
		double final_max_reproj_error = 2.0;	// max allowed reproject error after BA
		bool refine_with_ba = true;				// refine rotation and translation with BA
	};

	PairwiseTransformValidator(Options& options);

	bool acceptable(
		RelativeTransform& transform,
		std::shared_ptr<Camera> src_camera,
		std::shared_ptr<Camera> dst_camera,
		std::vector<Eigen::Vector2d>& src_observations,
		std::vector<Eigen::Vector2d>& dst_observations
	);

	bool acceptable(Scene& two_views_scene);

private:
	Options m_options;
};