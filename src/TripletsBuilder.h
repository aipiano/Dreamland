#pragma once

#include "EpipolarGraph.h"
#include "Triplets.h"

// find all triplets in a graph
class TripletsBuilder
{
public:
	void build(EpipolarGraph& epi_graph);
	void swap(Triplets& triplets);
private:
	inline void ensure_increasing_order(EpipolarGraph::edge_descriptor& e);
	Triplets m_triplets;
};
