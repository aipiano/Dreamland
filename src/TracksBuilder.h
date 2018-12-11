#pragma once

#include "FeaturePool.h"
#include "MatchPool.h"
#include "Tracks.h"
#include "EpipolarGraph.h"
#include <memory>
#include <map>
#include <unordered_map>

struct TrackHeader
{
	int view_id = -1;
	int feature_index = -1;

	TrackHeader(int view_id = -1, int feature_index = -1);

	bool operator==(const TrackHeader& other);
	friend bool operator==(const TrackHeader& left, const TrackHeader& right);
	friend bool operator<(const TrackHeader& left, const TrackHeader& right);
};

struct TrackHeaderHasher
{
	std::hash<int> int_hasher;
	inline std::size_t operator()(const TrackHeader& header) const;
};

class TracksBuilder
{
public:
	void build(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, PairwiseMatches& pairs);
	void build(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, EpipolarGraph& epi_graph);
	// 删除tracks中过短的或者view id有冲突的
	void filter(int min_length = 2, int max_length = 10);
	// 对每个track中的node进行排序，使node按照view_id升序排列
	void sort_nodes();
	void swap(Tracks& tracks);

protected:
	void unite(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, PairwiseMatch& pair);
	// 将union_map中的track转移到m_tracks中
	void move_union_map_to_tracks();
	bool view_id_collision(Track& track);

protected:
	typedef std::unordered_map<TrackHeader, std::shared_ptr<Track>, TrackHeaderHasher> UnionMap;
	typedef std::unordered_set<std::shared_ptr<Track>> TrackSet;
	//typedef std::map<TrackHeader, std::shared_ptr<Track>> UnionMap;
	Tracks m_tracks;
	UnionMap m_union_map;
	//std::set<std::shared_ptr<Track>> track_set;
	TrackSet m_track_set;
};
