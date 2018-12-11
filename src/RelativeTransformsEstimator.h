#pragma once

#include "FeaturePool.h"
#include "MatchPool.h"
#include "ViewCameraBinder.h"
#include "EpipolarGraph.h"
#include <memory>
#include <map>

class RelativeTransformsEstimator
{
public:
	virtual bool estimate(
		std::shared_ptr<FeaturePool> feature_pool, 
		std::shared_ptr<MatchPool> match_pool,
		ViewCameraBinder& view_camera_binder,
		EpipolarGraph& epi_graph	// must set RelativeTransforms
	) = 0;
};
