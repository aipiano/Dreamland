#pragma once

#include "EpipolarGraph.h"

bool reject_edge_by_rotation(EpipolarGraph& epi_graph, double max_rotation_error_in_degree = 5.0);