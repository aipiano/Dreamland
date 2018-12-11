#pragma once

#include "RelativeTransformsEstimator.h"
#include "EpipolarGraph.h"

// Do not support pano images.
class FivePointsEstimator final : public RelativeTransformsEstimator
{
public:
	struct Options
	{
		bool refine_with_ba = true;
		size_t min_num_inliers = 30;			// min inliers setisfy the init_max_reproj_error
		double ransac_threshold = 4.0;			// Sampson distance in pixels
		double init_max_reproj_error = 5.0;		// max allowed reproject error after triangulate
		double final_max_reproj_error = 2.0;	// max allowed reproject error after BA
	};

	FivePointsEstimator(Options& options);

	// Í¨¹ý RelativeTransformsEstimator ¼Ì³Ð
	virtual bool estimate(
		std::shared_ptr<FeaturePool> feature_pool, 
		std::shared_ptr<MatchPool> match_pool,
		ViewCameraBinder& view_camera_binder,
		EpipolarGraph & epi_graph
	) override;

private:
	inline void copy2extrinsic(cv::Matx31d& r, cv::Matx31d& t, CameraExtrinsic& rt);
	Options m_options;
};


