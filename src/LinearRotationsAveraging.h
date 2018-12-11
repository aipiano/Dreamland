#pragma once

#include "GlobalRotationsEstimator.h"

class LinearRotationsAveraging final : public GlobalRotationsEstimator
{
public:
	// Í¨¹ý GlobalRotationsEstimator ¼Ì³Ð
	virtual bool estimate(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, ViewCameraBinder & view_camera_binder, EpipolarGraph & epi_graph) override;
};
