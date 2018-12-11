#pragma once

#include <vector>

struct PairwiseMatch
{
	int id1 = -1;
	int id2 = -1;

	PairwiseMatch(){}

	PairwiseMatch(int id1, int id2)
		: id1(id1), id2(id2)
	{
	}

	bool operator==(const PairwiseMatch& other)
	{
		return (id1 == other.id1 && id2 == other.id2);
	}
	friend bool operator==(const PairwiseMatch& left, const PairwiseMatch& right);
	friend bool operator<(const PairwiseMatch& left, const PairwiseMatch& right);
};

struct PairwiseMatchHasher
{
	std::hash<int> int_hasher;
	inline std::size_t operator()(const PairwiseMatch& pair) const
	{
		size_t seed = int_hasher(pair.id1) + 0x9e3779b9;
		return seed ^ (int_hasher(pair.id2) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
	}
};

inline bool operator<(const PairwiseMatch& left, const PairwiseMatch& right)
{
	return left.id1 < right.id1 || (left.id1 == right.id1 && left.id2 < right.id2);
}

inline bool operator==(const PairwiseMatch& left, const PairwiseMatch& right)
{
	// Hash函数是有序的，所以比较函数也应该有序。方向无关性可通过保存时调整id的顺序实现
	return (left.id1 == right.id1 && left.id2 == right.id2);
}

typedef std::vector<PairwiseMatch> PairwiseMatches;
