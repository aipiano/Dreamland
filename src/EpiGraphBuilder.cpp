#include "EpiGraphBuilder.h"
#include "Utils.h"

#include <boost\graph\connected_components.hpp>
#include <vector>
#include <iostream>


using namespace Eigen;
using namespace boost;
using namespace std;

void EpiGraphBuilder::build(std::shared_ptr<MatchPool> match_pool, std::vector<int>& image_ids, int min_matches)
{
	m_graph.clear();
	size_t view_count = image_ids.size();
	for (size_t i = 0; i < view_count; ++i)
	{
		GlobalTransform transform;
		transform.is_inlier = true;
		transform.view_id = image_ids[i];
		transform.rt = CameraExtrinsic::Zero();

		add_vertex(transform, m_graph);
	}

	for (size_t i = 0; i < view_count; ++i)
	{
		int id_i = image_ids[i];
		for (size_t j = i + 1; j < view_count; ++j)
		{
			int id_j = image_ids[j];
			auto match_frame = match_pool->get_matches(id_i, id_j);
			if (match_frame == nullptr || match_frame->matches.size() < min_matches)
				continue;

			RelativeTransform transform;
			transform.rt = CameraExtrinsic::Zero();
			transform.is_inlier = true;

			add_edge(i, j, transform, m_graph);
		}
	}
}

void EpiGraphBuilder::filter(int min_degree)
{
	bool need_to_remove = false;
	size_t n = num_vertices(m_graph);
	for (size_t i = 0; i < n; ++i)
	{
		if (degree(i, m_graph) < min_degree)
		{
			m_graph[i].is_inlier = false;
		}
	}
}

void EpiGraphBuilder::keep_largest_cc()
{
	size_t vertices_count = num_vertices(m_graph);
	size_t edges_count = num_edges(m_graph);
	cout << "\n#views = " << vertices_count << ", #edges = " << edges_count << endl;

	// remove all outlier edges and vertices in the graph
	for (size_t i = 0; i < vertices_count; ++i)
	{
		if (!m_graph[i].is_inlier)
			clear_vertex(i, m_graph);
	}
	remove_edge_if([](EpipolarGraph::edge_descriptor& e) 
	{
		if (((RelativeTransform*)e.m_eproperty)->is_inlier)
			return false;
		cout << "Remove edge " << e.m_source << " - " << e.m_target << endl;
		return true;
	}, m_graph);
	

	vector<size_t> component(vertices_count);
	size_t num = connected_components(m_graph, component.data());
	if (num <= 1)
	{
		cout << "After removing outliers: " << endl;
		cout << "#views = " << vertices_count << ", #edges = " << num_edges(m_graph) << endl;
		return;
	}

	vector<size_t> hist(num);
	for (size_t i = 0; i < vertices_count; ++i)
	{
		++hist[component[i]];
	}

	//find largest connected component
	size_t max_vertices = hist[0];
	size_t cc_id = 0;
	for (int i = 1; i < num; ++i)
	{
		if (hist[i] > max_vertices)
		{
			max_vertices = hist[i];
			cc_id = i;
		}
	}

	//only keep the largest connected component
	EpipolarGraph graph_cc;
	for (size_t i = 0; i < vertices_count; ++i)
	{
		if (component[i] != cc_id)
			continue;

		add_vertex(m_graph[i], graph_cc);
	}

	EpipolarGraph::edge_iterator e_begin, e_end;
	tie(e_begin, e_end) = edges(m_graph);
	for (auto it_e = e_begin; it_e != e_end; ++it_e)
	{
		auto u = it_e->m_source;
		auto v = it_e->m_target;
		if (component[u] != component[v])
			continue;
		if (component[u] != cc_id)
			continue;
		add_edge(u, v, m_graph[*it_e], graph_cc);
	}
	m_graph.swap(graph_cc);

	size_t prev_count = vertices_count;
	vertices_count = num_vertices(m_graph);
	cout << "After keeping largest connected commponent: " << endl;
	cout << "#views = " << vertices_count << ", #edges = " << num_edges(m_graph) << endl;
	cout << "Removed " << prev_count - vertices_count << " vertices." << endl;
}

void EpiGraphBuilder::swap(EpipolarGraph & graph)
{
	m_graph.swap(graph);
}
