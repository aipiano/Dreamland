#pragma once

#include <list>
#include <vector>
#include <Eigen\Eigen>

struct TrackNode
{
	Eigen::Vector2d observation;
	int view_id = -1;
	bool is_inlier = false;

	TrackNode() {}
	TrackNode(Eigen::Vector2d& observation, int view_id, bool is_inlier = true)
		: observation(observation), view_id(view_id), is_inlier(is_inlier)
	{
	}
};

class Track
{
public:
	Track();
	Track(const Track& other);
	Track(Track&& other);
	Track& operator=(const Track& other);
	void operator=(Track&& other);

public:
	Eigen::Vector3d point3d;
	std::vector<TrackNode> nodes;

	bool is_inlier = false;
};

typedef std::list<Track> Tracks;
