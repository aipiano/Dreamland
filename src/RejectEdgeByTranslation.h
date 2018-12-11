#pragma once

#include "EpipolarGraph.h"

bool reject_edge_by_translation(EpipolarGraph& epi_graph, double bad_edge_tolerance = 0.08, int num_directions = 48);
