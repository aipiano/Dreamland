#include "Tracks.h"

Track::Track()
{
}

Track::Track(const Track & other)
{
	point3d = other.point3d;
	nodes = other.nodes;
	is_inlier = other.is_inlier;
}

Track::Track(Track && other)
{
	point3d = other.point3d;
	nodes = std::move(other.nodes);
	is_inlier = other.is_inlier;
}

Track & Track::operator=(const Track & other)
{
	point3d = other.point3d;
	nodes = other.nodes;
	is_inlier = other.is_inlier;

	return *this;
}

void Track::operator=(Track && other)
{
	point3d = other.point3d;
	nodes = std::move(other.nodes);
	is_inlier = other.is_inlier;
}
