#pragma once

#include <memory>
#include "FeaturePool.h"
#include "MatchPool.h"
#include "EpipolarGraph.h"

class EpiGraphBuilder
{
public:
	void build(std::shared_ptr<MatchPool> match_pool, std::vector<int>& image_ids, int min_matches = 50);
	// ���ȹ�С�Ķ�����Ϊoutlier
	void filter(int min_degree = 3);
	// ɾ�����б����Ϊoutlier�ıߺͶ��㣬������ͼ�е������ͨ��ͼ
	void keep_largest_cc();
	
	void swap(EpipolarGraph& graph);

protected:
	EpipolarGraph m_graph;
};