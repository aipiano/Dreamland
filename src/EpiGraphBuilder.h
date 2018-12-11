#pragma once

#include <memory>
#include "FeaturePool.h"
#include "MatchPool.h"
#include "EpipolarGraph.h"

class EpiGraphBuilder
{
public:
	void build(std::shared_ptr<MatchPool> match_pool, std::vector<int>& image_ids, int min_matches = 50);
	// 将度过小的顶点标记为outlier
	void filter(int min_degree = 3);
	// 删除所有被标记为outlier的边和顶点，并保留图中的最大连通子图
	void keep_largest_cc();
	
	void swap(EpipolarGraph& graph);

protected:
	EpipolarGraph m_graph;
};