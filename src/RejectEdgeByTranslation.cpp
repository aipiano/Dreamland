#include "RejectEdgeByTranslation.h"
#include "Utils.h"
#include <vector>
#include <Eigen\Eigen>
#include <random>

using namespace std;
using namespace Eigen;
using namespace boost;

typedef pair<size_t, size_t> DiEdge;

struct MFASNode
{
	// using unordered_map is another good choice.

	vector<pair<size_t, double>> incoming_nodes;	
	vector<pair<size_t, double>> outgoing_nodes;
	double incoming_weight = 0;
	double outgoing_weight = 0;
};

void build_diedges(EpipolarGraph & epi_graph, vector<DiEdge>& diedges, MatrixXd& translations)
{
	auto edge_count = num_edges(epi_graph);
	translations.resize(edge_count, 3);
	diedges.clear();
	diedges.reserve(edge_count);

	EpipolarGraph::edge_iterator e_begin, e_end;
	tie(e_begin, e_end) = edges(epi_graph);
	size_t edge_idx = 0;
	for (auto it_e = e_begin; it_e != e_end; ++it_e, ++edge_idx)
	{
		auto i = it_e->m_source;
		auto j = it_e->m_target;
		if (i > j)	//»Ù i > j ‘ÚΩªªª
		{
			i = i^j;
			j = i^j;
			i = i^j;
		}

		diedges.emplace_back(i, j);

		// transform translation to the world coordinate
		Matrix3d Rj;
		angle_axis_to_rotation_matrix(epi_graph[j].rt.topRows(3), Rj);
		translations.row(edge_idx).noalias() = (Rj.transpose() * epi_graph[*it_e].rt.bottomRows(3)).transpose();
	}
}

void sample_directions(MatrixXd& translations, vector<Vector3d>& directions, int num_directions)
{
	Vector3d mean = (translations.colwise().sum()).transpose() / translations.rows();
	MatrixXd deltas = translations.rowwise() - mean.transpose();
	Vector3d var = (deltas.colwise().squaredNorm()).transpose() / (translations.rows() - 1);
	Vector3d sigma = var.cwiseSqrt();

	std::random_device rd;
	std::mt19937 engine(rd());
	std::normal_distribution<double> dist_x(mean[0], sigma[0]);
	std::normal_distribution<double> dist_y(mean[1], sigma[1]);
	std::normal_distribution<double> dist_z(mean[2], sigma[2]);

	directions.clear();
	directions.reserve(num_directions);
	for (int i = 0; i < num_directions; ++i)
	{
		directions.emplace_back(dist_x(engine), dist_y(engine), dist_z(engine));
		directions.back().normalize();
	}
}

void flip_neg_edges(vector<DiEdge>& diedges, VectorXd& weights, vector<DiEdge>& fliped_diedges)
{
	fliped_diedges = diedges;
	for (size_t i = 0; i < diedges.size(); ++i)
	{
		if (weights[i] >= 0) continue;
		swap(fliped_diedges[i].first, fliped_diedges[i].second);
		weights[i] *= -1;
	}
}

void find_mfas_order(vector<DiEdge>& diedges, VectorXd& weights, size_t vertex_count, vector<size_t>& order)
{
	vector<MFASNode> mfas_nodes(vertex_count);	// using unordered_map is another good choice.
	for (size_t edge_idx = 0; edge_idx < diedges.size(); ++edge_idx)
	{
		auto i = diedges[edge_idx].first;
		auto j = diedges[edge_idx].second;
		auto w = weights[edge_idx];

		mfas_nodes[j].incoming_weight += w;
		mfas_nodes[i].outgoing_weight += w;
		mfas_nodes[j].incoming_nodes.emplace_back(i, w);
		mfas_nodes[i].outgoing_nodes.emplace_back(j, w);
	}

	order.clear();
	order.reserve(vertex_count);
	vector<bool> chosed(vertex_count, false);
	while (order.size() < vertex_count)
	{
		size_t choice = 0;
		double max_score = 0;

		for (int i = 0; i < vertex_count; ++i)
		{
			if (chosed[i]) continue;
			if (mfas_nodes[i].incoming_weight < 1e-8)	// this is a source
			{
				choice = i;
				break;
			}
			else
			{
				double score = (mfas_nodes[i].outgoing_weight + 1) / (mfas_nodes[i].incoming_weight + 1);
				if (score > max_score) 
				{
					max_score = score;
					choice = i;
				}
			}
		}

		// update mfas graph
		auto& incoming_nodes = mfas_nodes[choice].incoming_nodes;
		auto& outgoing_nodes = mfas_nodes[choice].outgoing_nodes;
		for (auto it_in = incoming_nodes.begin(); it_in != incoming_nodes.end(); ++it_in)
			mfas_nodes[it_in->first].outgoing_weight -= it_in->second;
		for (auto it_out = outgoing_nodes.begin(); it_out != outgoing_nodes.end(); ++it_out)
			mfas_nodes[it_out->first].incoming_weight -= it_out->second;

		order.push_back(choice);
		chosed[choice] = true;
	}
}

void get_absolute_distance(vector<size_t>& order, vector<size_t>& absolute_distance)
{
	for (size_t i = 0; i < order.size(); ++i)
		absolute_distance[order[i]] = i;
}

void remove_bad_edges(EpipolarGraph & epi_graph, vector<double>& broken_weights, double broken_threshold)
{
	EpipolarGraph::edge_iterator e_begin, e_end;
	tie(e_begin, e_end) = edges(epi_graph);
	size_t edge_idx = 0;
	for (auto it_e = e_begin; it_e != e_end; ++it_e, ++edge_idx)
	{
		//cout << broken_weights[edge_idx] << endl;
		if (broken_weights[edge_idx] < broken_threshold)
			continue;
		epi_graph[*it_e].is_inlier = false;
	}
}

bool reject_edge_by_translation(EpipolarGraph & epi_graph, double bad_edge_tolerance, int num_directions)
{
	double broken_threshold = bad_edge_tolerance * num_directions;
	size_t vertex_count = num_vertices(epi_graph);

	// globally visiable to all threads
	vector<DiEdge> diedges;
	MatrixXd translations;
	vector<Vector3d> directions;

	build_diedges(epi_graph, diedges, translations);
	sample_directions(translations, directions, num_directions);

	vector<double> broken_weights(diedges.size(), 0.0);

	// thread's private members
	VectorXd weights(diedges.size());
	vector<DiEdge> fliped_edges(diedges.size());
	vector<size_t> order(vertex_count);
	vector<size_t> absolute_distance(vertex_count);

	// TODO: parallel
	for (size_t i = 0; i < num_directions; ++i)
	{
		// 'translations' is a n rows by 3 cols matrix, where n is the number of edges
		weights.noalias() = translations * directions[i];

		flip_neg_edges(diedges, weights, fliped_edges);
		find_mfas_order(fliped_edges, weights, vertex_count, order);

		get_absolute_distance(order, absolute_distance);

		// accumulate broken weights
		for (size_t edge_idx = 0; edge_idx < fliped_edges.size(); ++edge_idx)
		{
			auto& diedge = fliped_edges[edge_idx];
			size_t xi = absolute_distance[diedge.first];
			size_t xj = absolute_distance[diedge.second];
			if ((xj - xi) * weights[edge_idx] < 0)
			{
				// #pragma omp critical
				broken_weights[edge_idx] += weights[edge_idx];
			}
		}
	}

	remove_bad_edges(epi_graph, broken_weights, broken_threshold);

	return true;
}
