#pragma once

#include <vector>

struct TripletMatch
{
	int id1 = -1;
	int id2 = -1;
	int id3 = -1;

	TripletMatch(int id1, int id2, int id3)
		: id1(id1), id2(id2), id3(id3)
	{}

	bool operator==(const TripletMatch& other)
	{
		return (id1 == other.id1 && id2 == other.id2 && id3 == other.id3);
	}
	friend bool operator==(const TripletMatch& left, const TripletMatch& right);
	friend bool operator<(const TripletMatch& left, const TripletMatch& right);
};

inline bool operator<(const TripletMatch& left, const TripletMatch& right)
{
	return left.id1 < right.id1 || 
		(left.id1 == right.id1 && left.id2 < right.id2) || 
		(left.id1 == right.id1 && left.id2 == right.id2 && left.id3 < right.id3);
}

inline bool operator==(const TripletMatch& left, const TripletMatch& right)
{
	// Hash函数是有序的，所以比较函数也应该有序。方向无关性可通过保存时调整id的顺序实现
	return (left.id1 == right.id1 && left.id2 == right.id2 && left.id3 == right.id3);
}

typedef std::vector<TripletMatch> TripletMatches;