#include "TracksBuilder.h"
#include <Eigen/Eigen>
#include <set>
#include <unordered_set>

using namespace std;
using namespace boost;
using namespace Eigen;

void TracksBuilder::build(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, PairwiseMatches & pairs)
{
	m_union_map.clear();
	for (auto& pair : pairs)
	{
		unite(feature_pool, match_pool, pair);
	}
	move_union_map_to_tracks();
}

void TracksBuilder::build(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, EpipolarGraph & epi_graph)
{
	m_union_map.clear();
	EpipolarGraph::edge_iterator e_begin, e_end;
	tie(e_begin, e_end) = edges(epi_graph);
	for (auto it_e = e_begin; it_e != e_end; ++it_e)
	{
		if (!epi_graph[*it_e].is_inlier)
			continue;
		
		int view_id1 = epi_graph[it_e->m_source].view_id;
		int view_id2 = epi_graph[it_e->m_target].view_id;
		
		unite(feature_pool, match_pool, PairwiseMatch(view_id1, view_id2));
	}
	move_union_map_to_tracks();
}

void TracksBuilder::unite(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, PairwiseMatch & pair)
{
	const auto match_frame = match_pool->get_matches(pair);
	if (match_frame == nullptr) return;

	// hold pointers to add reference count
	auto train_kps_ptr = feature_pool->get_keypoints(match_frame->train_id);
	auto query_kps_ptr = feature_pool->get_keypoints(match_frame->query_id);
	const auto& train_kps = *train_kps_ptr;
	const auto& query_kps = *query_kps_ptr;
	auto& matches = match_frame->matches;

	TrackHeader train_header(match_frame->train_id, -1);
	TrackHeader query_header(match_frame->query_id, -1);
	for (size_t i = 0; i < matches.size(); ++i)
	{ 
		int train_idx = train_header.feature_index = matches[i].trainIdx;
		int query_idx = query_header.feature_index = matches[i].queryIdx;

		Vector2d train_observation(train_kps[train_idx].pt.x, train_kps[train_idx].pt.y);
		Vector2d query_observation(query_kps[query_idx].pt.x, query_kps[query_idx].pt.y);

		auto& train_track_ptr = m_union_map[train_header];
		auto& query_track_ptr = m_union_map[query_header];
		if (train_track_ptr == nullptr && query_track_ptr == nullptr)
		{	// 双方均无链，则新建一条以这两方为节点的链
			auto new_track = make_shared<Track>();
			new_track->is_inlier = true;
			new_track->nodes.push_back(TrackNode(train_observation, train_header.view_id));
			new_track->nodes.push_back(TrackNode(query_observation, query_header.view_id));

			train_track_ptr = new_track;
			query_track_ptr = new_track;
		}
		else if (train_track_ptr != nullptr && query_track_ptr == nullptr)
		{	// 无链的一方并入有链的一方
			train_track_ptr->nodes.push_back(TrackNode(query_observation, query_header.view_id));
			query_track_ptr = train_track_ptr;
		}
		else if (train_track_ptr == nullptr && query_track_ptr != nullptr)
		{	// 无链的一方并入有链的一方
			query_track_ptr->nodes.push_back(TrackNode(train_observation, train_header.view_id));
			train_track_ptr = query_track_ptr;
		}
		// 其他情况对应于有冲突的情况，直接忽略即可
	}
}

void TracksBuilder::move_union_map_to_tracks()
{
	m_tracks.clear();
	m_track_set.clear();
	for (auto it_track = m_union_map.begin(); it_track != m_union_map.end(); ++it_track)
	{
		if (it_track->second == nullptr) continue;
		m_track_set.insert(it_track->second);
	}

	for (auto it_track = m_track_set.begin(); it_track != m_track_set.end(); ++it_track)
	{
		m_tracks.push_back(std::move(**it_track));
	}
	m_track_set.clear();
	m_union_map.clear();
}

bool TracksBuilder::view_id_collision(Track & track)
{
	auto& nodes = track.nodes;
	set<int> id_set;

	for (auto& node : nodes)
	{
		if (id_set.find(node.view_id) != id_set.end())
			return true;	// the view_id arready in the set

		id_set.insert(node.view_id);
	}

	return false;
}

void TracksBuilder::filter(int min_length, int max_length)
{
	for (auto it_track = m_tracks.begin(); it_track != m_tracks.end();)
	{
		int track_length = it_track->nodes.size();
		// track长度过短或过长或存在id冲突，则删除，id冲突的track一般是在match阶段通过了cross test的outlier
		if (track_length < min_length || track_length > max_length || view_id_collision(*it_track))
		{
			it_track = m_tracks.erase(it_track);
		}
		else
		{
			++it_track;
		}
	}
}

void TracksBuilder::sort_nodes()
{
	for (auto it_track = m_tracks.begin(); it_track != m_tracks.end(); ++it_track)
	{
		auto& nodes = it_track->nodes;
		std::sort(nodes.begin(), nodes.end(), [](TrackNode& a, TrackNode& b) {
			return a.view_id < b.view_id;
		});
	}
}

void TracksBuilder::swap(Tracks& tracks)
{
	m_tracks.swap(tracks);
}

inline bool operator==(const TrackHeader & left, const TrackHeader & right)
{
	return (left.view_id == right.view_id) && (left.feature_index == right.feature_index);
}

inline bool operator<(const TrackHeader & left, const TrackHeader & right)
{
	return left.view_id < right.view_id || (left.view_id == right.view_id && left.feature_index < right.feature_index);
}

TrackHeader::TrackHeader(int view_id, int feature_index)
	:view_id(view_id), feature_index(feature_index)
{
}

bool TrackHeader::operator==(const TrackHeader & other)
{
	return (view_id == other.view_id) && (feature_index == other.feature_index);
}

std::size_t TrackHeaderHasher::operator()(const TrackHeader & header) const
{
	size_t seed = int_hasher(header.view_id) + 0x9e3779b9;
	return seed ^ (int_hasher(header.feature_index) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
