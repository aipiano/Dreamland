#pragma once

#include "EpipolarGraph.h"
#include "MatchPool.h"
#include "FeaturePool.h"
#include "ViewCameraBinder.h"

// ��С�������λ��������֧��ȫ��������������������������㹲������
void refine_relative_translations_with_known_rotations(
	EpipolarGraph& epi_graph, 
	std::shared_ptr<FeaturePool> feature_pool, 
	std::shared_ptr<MatchPool> match_pool,
	ViewCameraBinder& view_camera_binder
);
