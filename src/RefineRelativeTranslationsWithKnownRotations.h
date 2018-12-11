#pragma once

#include "EpipolarGraph.h"
#include "MatchPool.h"
#include "FeaturePool.h"
#include "ViewCameraBinder.h"

// 最小二乘求解位移向量，支持全景相机，但不适用于所有特征点共面的情况
void refine_relative_translations_with_known_rotations(
	EpipolarGraph& epi_graph, 
	std::shared_ptr<FeaturePool> feature_pool, 
	std::shared_ptr<MatchPool> match_pool,
	ViewCameraBinder& view_camera_binder
);
