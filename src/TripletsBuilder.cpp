#include "TripletsBuilder.h"
#include <vector>

using namespace std;

void TripletsBuilder::build(EpipolarGraph & epi_graph)
{
	m_triplets.clear();

	size_t n = num_vertices(epi_graph);
	vector<size_t> intersection(n, 0);
	vector<EpipolarGraph::edge_descriptor> edge_ik(n);

	EpipolarGraph::edge_iterator e_begin, e_end;
	EpipolarGraph::out_edge_iterator oe_begin, oe_end;

	size_t id = 1;
	tie(e_begin, e_end) = edges(epi_graph);
	for (auto it_e_ij = e_begin; it_e_ij != e_end; ++it_e_ij)
	{
		auto i = it_e_ij->m_source;
		auto j = it_e_ij->m_target;
		if (i > j)	//若 i > j 则交换
		{
			i = i^j;
			j = i^j;
			i = i^j;
		}

		tie(oe_begin, oe_end) = out_edges(i, epi_graph);
		for (auto it_oe = oe_begin; it_oe != oe_end; ++it_oe)
		{
			size_t k = it_oe->m_target;
			intersection[k] = id;
			edge_ik[k] = *it_oe;
		}

		tie(oe_begin, oe_end) = out_edges(j, epi_graph);
		for (auto it_oe = oe_begin; it_oe != oe_end; ++it_oe)
		{
			//不相交，或者之前已经枚举过，则继续
			//所得triplet的三个顶点索引一定有i < j < k， 所以如果k <= j，则该triplet一定在之前就被枚举过
			size_t k = it_oe->m_target;
			if (intersection[k] != id || k <= j)
				continue;

			Triplet triplet;
			triplet.e_ij = *it_e_ij;
			triplet.e_ik = edge_ik[k];
			triplet.e_jk = *it_oe;

			// i < j < k, for each edge, we have m_source < m_target
			ensure_increasing_order(triplet.e_ij);
			ensure_increasing_order(triplet.e_ik);
			ensure_increasing_order(triplet.e_jk);

			m_triplets.push_back(triplet);
		}
		++id;
	}
}

void TripletsBuilder::swap(Triplets & triplets)
{
	m_triplets.swap(triplets);
}

inline void TripletsBuilder::ensure_increasing_order(EpipolarGraph::edge_descriptor & e)
{
	if (e.m_source > e.m_target)
		std::swap(e.m_source, e.m_target);
}
