#include "RejectEdgeByRotation.h"
#include "Utils.h"
#include "Triplets.h"
#include "TripletsBuilder.h"

using namespace Eigen;
using namespace std;

bool reject_edge_by_rotation(EpipolarGraph & epi_graph, double max_rotation_error_in_degree)
{
	EpipolarGraph::edge_iterator e_begin, e_end;
	tie(e_begin, e_end) = edges(epi_graph);
	for (auto e_it = e_begin; e_it != e_end; ++e_it)
	{
		((RelativeTransform*)e_it->m_eproperty)->is_inlier = false;
	}

	Triplets triplets;
	TripletsBuilder triplet_builder;
	triplet_builder.build(epi_graph);
	triplet_builder.swap(triplets);
	if (triplets.size() < 1)
	{
		cout << "No triplet in graph. Maybe the scene is too tiny or views are too different to detect stable relationships." << endl;
		return false;
	}

	Matrix3d R_err;
	Vector3d r_err;
	double max_err_rad = deg2rad(max_rotation_error_in_degree);
	for (auto it_triplet = triplets.begin(); it_triplet != triplets.end(); ++it_triplet)
	{
		RelativeTransform& transform_ij = *(RelativeTransform*)it_triplet->e_ij.m_eproperty;
		RelativeTransform& transform_ik = *(RelativeTransform*)it_triplet->e_ik.m_eproperty;
		RelativeTransform& transform_jk = *(RelativeTransform*)it_triplet->e_jk.m_eproperty;
		// must have i < j < k
		auto i = it_triplet->e_ij.m_source;
		auto j = it_triplet->e_ij.m_target;
		int view_i = epi_graph[i].view_id;
		int view_j = epi_graph[j].view_id;

		Matrix3d Rij, Rik, Rjk;
		angle_axis_to_rotation_matrix(transform_ij.rt.topRows(3), Rij);
		angle_axis_to_rotation_matrix(transform_ik.rt.topRows(3), Rik);
		angle_axis_to_rotation_matrix(transform_jk.rt.topRows(3), Rjk);

		R_err = Rij * Rjk * Rik.transpose();
		rotation_matrix_to_angle_axis(R_err, r_err);
		if (r_err.norm() < max_err_rad)
		{
			transform_ij.is_inlier = transform_ik.is_inlier = transform_jk.is_inlier = true;
			//++it_triplet;
		}
		/*else
		{
		it_triplet = triplets.erase(it_triplet);
		}*/
	}

	return true;
}
