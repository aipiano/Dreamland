#pragma once

#include "FeaturePool.h"
#include "MatchPool.h"
#include "ViewCameraBinder.h"
#include "EpipolarGraph.h"
#include <memory>

class GlobalTranslationsEstimator
{
public:
	virtual bool estimate(
		std::shared_ptr<FeaturePool> feature_pool, 
		std::shared_ptr<MatchPool> match_pool, 
		ViewCameraBinder& view_camera_binder,
		EpipolarGraph& epi_graph
	) = 0;
};
