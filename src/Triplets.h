#pragma once

#include "EpipolarGraph.h"
#include <list>

struct Triplet
{
	EpipolarGraph::edge_descriptor e_ij;
	EpipolarGraph::edge_descriptor e_ik;
	EpipolarGraph::edge_descriptor e_jk;

	Triplet() {};
	Triplet(
		EpipolarGraph::edge_descriptor e_ij, EpipolarGraph::edge_descriptor e_ik, EpipolarGraph::edge_descriptor e_jk)
		: e_ij(e_ij), e_ik(e_ik), e_jk(e_jk)
	{}
};

typedef std::vector<Triplet> Triplets;