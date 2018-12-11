#pragma once

#include <boost/config.hpp> // put this first to suppress some VC++ warnings
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

#include "GlobalTransform.h"
#include "RelativeTransform.h"

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, GlobalTransform, RelativeTransform> EpipolarGraph;