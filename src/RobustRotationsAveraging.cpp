#include "RobustRotationsAveraging.h"
#include "Utils.h"
#include "LinearRotationsAveraging.h"
#include <Eigen\Sparse>
#include <ClpInterior.hpp>
#include <ClpSimplex.hpp>
#include <CoinBuild.hpp>
#include <CoinPackedVector.hpp>

using namespace std;
using namespace boost;
using namespace Eigen;

RobustRotationsAveraging::RobustRotationsAveraging(Options options)
	:m_options(options)
{
}

bool RobustRotationsAveraging::estimate(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, ViewCameraBinder & view_camera_binder, EpipolarGraph & epi_graph)
{
	vector<Matrix3d> R, Rij;
	SparseMatrix<double, RowMajor> A;
	initialize_system(epi_graph, R, Rij, A);

	// L1 averaging. Output to R.
	if (!l1_averaging(epi_graph, Rij, A, R))
		return false;

	// IRLS averaging. Output to r.
	if (!irls_averaging(epi_graph, Rij, A, R))
		return false;

	// save to epipolar graph
	update_graph(epi_graph, R);
	
	filter_outlier_edges(epi_graph, Rij);

	return true;
}

void RobustRotationsAveraging::initialize_system(
	const EpipolarGraph & epi_graph,
	std::vector<Eigen::Matrix3d>& R, 
	std::vector<Eigen::Matrix3d>& Rij, 
	Eigen::SparseMatrix<double, RowMajor>& A
)
{
	int num_vertex = num_vertices(epi_graph);
	int num_edge = num_edges(epi_graph);

	R.clear();
	R.resize(num_vertex, Matrix3d::Identity());

	Rij.clear();
	Rij.resize(num_edge);

	A = SparseMatrix<double, RowMajor>(3 * num_edge, 3 * num_vertex);
	A.reserve(VectorXi::Constant(A.rows(), 2));

	EpipolarGraph::edge_iterator e_begin, e_end;
	tie(e_begin, e_end) = edges(epi_graph);
	size_t edge_idx = 0;
	for (auto it_e = e_begin; it_e != e_end; ++it_e, ++edge_idx)
	{
		size_t i = it_e->m_source;
		size_t j = it_e->m_target;
		if (i > j)	//若 i > j 则交换
		{
			i = i^j;
			j = i^j;
			i = i^j;
		}

		// build Rij and r_ij
		RelativeTransform& transform = *((RelativeTransform*)it_e->m_eproperty);
		angle_axis_to_rotation_matrix(transform.rt.topRows(3), Rij[edge_idx]);
		/*if (epi_graph[i].view_id != transform.src_id)
		{
			Rij[edge_idx].transposeInPlace();
		}*/

		// build A
		int start_row = edge_idx * 3;
		int start_col_i = i * 3;
		int start_col_j = j * 3;
		A.insert(start_row, start_col_i) = -1.0;
		A.insert(start_row, start_col_j) = 1.0;
		A.insert(start_row + 1, start_col_i + 1) = -1.0;
		A.insert(start_row + 1, start_col_j + 1) = 1.0;
		A.insert(start_row + 2, start_col_i + 2) = -1.0;
		A.insert(start_row + 2, start_col_j + 2) = 1.0;
	}
	A.makeCompressed();
}

void RobustRotationsAveraging::calc_rotation_errors(
	const EpipolarGraph & epi_graph, 
	const std::vector<Eigen::Matrix3d>& R, 
	const std::vector<Eigen::Matrix3d>& Rij, 
	Eigen::VectorXd& delta_r_ij
)
{
	delta_r_ij.resize(3 * Rij.size());

	EpipolarGraph::edge_iterator e_begin, e_end;
	tie(e_begin, e_end) = edges(epi_graph);
	size_t edge_idx = 0;
	Matrix3d delta_Rij;

	for (auto it_e = e_begin; it_e != e_end; ++it_e, ++edge_idx)
	{
		size_t i = it_e->m_source;
		size_t j = it_e->m_target;
		if (i > j)	//if i > j, then swap them
		{
			i = i^j;
			j = i^j;
			i = i^j;
		}

		delta_Rij = R[j].transpose() * Rij[edge_idx] * R[i];
		rotation_matrix_to_angle_axis(delta_Rij, delta_r_ij.middleRows(edge_idx * 3, 3));
	}
}

void RobustRotationsAveraging::update_global_rotations(
	const Eigen::VectorXd & delta_r,
	std::vector<Eigen::Matrix3d>& R
)
{
	Matrix3d delta_Rij;
	for (size_t i = 0; i < R.size(); ++i)
	{
		angle_axis_to_rotation_matrix(delta_r.middleRows(i * 3, 3), delta_Rij);
		R[i] *= delta_Rij;
	}
}

bool RobustRotationsAveraging::l1_averaging(
	const EpipolarGraph & epi_graph,
	const std::vector<Eigen::Matrix3d>& Rij,
	const Eigen::SparseMatrix<double, RowMajor>& A,
	std::vector<Eigen::Matrix3d>& R
)
{
	VectorXd delta_r_ij(3 * Rij.size());
	VectorXd delta_r(3 * R.size());
	double delta_norm = DBL_MAX;
	double prev_norm = DBL_MAX;
	int iter_count = 0;

	do
	{
		calc_rotation_errors(epi_graph, R, Rij, delta_r_ij);
		// solve A * delta_r = delta_r_ij;
		if (!solve_l1(A, delta_r_ij, delta_r))
			return false;
		update_global_rotations(delta_r, R);

		prev_norm = delta_norm;
		delta_norm = delta_r.norm();
		cout << "L1 Delta: " << delta_norm << endl;
		if (prev_norm < delta_norm)
			break;

		iter_count++;
	} while (delta_norm > 1e-2 && iter_count < 32);

	return true;
}

bool RobustRotationsAveraging::solve_l1(
	const Eigen::SparseMatrix<double, RowMajor>& A, 
	const Eigen::VectorXd & delta_r_ij, 
	Eigen::VectorXd & delta_r
)
{
	CoinBuild lp_builder;
	static std::array<double, 3> row_nag{ -1, 1, -1 };
	static std::array<double, 3> row_pos{ 1, -1, -1 };

	vector<int> inner_indices(A.innerIndexPtr(), A.innerIndexPtr() + A.outerSize() * 2);

	// min  1*s
	// s.t. [ A | -I][x] <= [ b]
	//		[-A | -I][s]		[-b]
	int delta_r_rows = delta_r.rows();
	for (size_t i = 0; i < A.rows(); ++i)
	{
		std::array<int, 3> indices{ inner_indices[2*i], inner_indices[2*i + 1], delta_r_rows + i };
		lp_builder.addRow(3, indices.data(), row_nag.data(), -COIN_DBL_MAX, delta_r_ij[i]);
		lp_builder.addRow(3, indices.data(), row_pos.data(), -COIN_DBL_MAX, -delta_r_ij[i]);
	}
	
	ClpSimplex lp_solver;
	int variable_count = delta_r_rows + delta_r_ij.rows();
	lp_solver.resize(0, variable_count);
	for (int x = 0; x < delta_r_rows; ++x)
	{
		lp_solver.setObjCoeff(x, 0);
		lp_solver.setColLower(x, -COIN_DBL_MAX);
		lp_solver.setColUpper(x, COIN_DBL_MAX);
	}
	for (int s = delta_r_rows; s < variable_count; ++s)
	{
		lp_solver.setObjCoeff(s, 1);
		lp_solver.setColLower(s, -COIN_DBL_MAX);
		lp_solver.setColUpper(s, COIN_DBL_MAX);
	}
	// fix the first view
	for (int i = 0; i < 3; ++i)
	{
		lp_solver.setColLower(i, 0);
		lp_solver.setColUpper(i, 0);
	}

	lp_solver.addRows(lp_builder);
	lp_solver.setLogLevel(0);
	lp_solver.dual();

	if (lp_solver.status() == 0)
	{
		delta_r = Map<VectorXd>(const_cast<double*>(lp_solver.getColSolution()), delta_r_rows);
		return true;
	}
	
	return false;
}

bool RobustRotationsAveraging::irls_averaging(const EpipolarGraph & epi_graph, const std::vector<Eigen::Matrix3d>& Rij, const Eigen::SparseMatrix<double, Eigen::RowMajor>& A, std::vector<Eigen::Matrix3d>& R)
{
	VectorXd delta_r_ij(3 * Rij.size());
	VectorXd delta_r = VectorXd::Zero(3 * R.size());
	VectorXd residual;
	VectorXd weights;
	SparseMatrix<double, RowMajor> AtW;

	// SparseQR is much more stable for least-square problem.
	SparseQR<SparseMatrix<double, RowMajor>, COLAMDOrdering<int>> solver;
	solver.analyzePattern(A.transpose() * A);
	if (solver.info() != Eigen::Success)
	{
		cout << "Analyze pattern failed." << endl;
		return false;
	}

	int iter_count = 0;
	double delta_norm = DBL_MAX;
	do
	{
		calc_rotation_errors(epi_graph, R, Rij, delta_r_ij);
		residual = A * delta_r - delta_r_ij;
		build_weights(residual, weights);

		AtW = A.transpose() * weights.asDiagonal();
		solver.factorize(AtW * A);
		if (solver.info() != Eigen::Success)
		{
			cout << "Factorize failed." << endl;
			return false;
		}

		delta_r = solver.solve(AtW * delta_r_ij);
		if (solver.info() != Eigen::Success)
		{
			cout << "Solve failed." << endl;
			return false;
		}

		update_global_rotations(delta_r, R);

		delta_norm = delta_r.norm();
		cout << "IRLS Delta: " << delta_norm << endl;
		iter_count++;

	} while (delta_norm > 1e-3 && iter_count < 32);

	return true;
}

void RobustRotationsAveraging::update_graph(EpipolarGraph & epi_graph, const std::vector<Eigen::Matrix3d>& R)
{
	// save to epipolar graph
	size_t vertices_count = num_vertices(epi_graph);
	Matrix3d base_R = R[0].transpose();
	Matrix3d rotated_R;
	for (size_t i = 0; i < vertices_count; ++i)
	{
		rotated_R = R[i] * base_R;
		rotation_matrix_to_angle_axis(rotated_R, epi_graph[i].rt.topRows(3));
	}
}

void RobustRotationsAveraging::build_weights(const Eigen::VectorXd& residuals, Eigen::VectorXd& weights)
{
	double d2 = m_options.sigma * m_options.sigma;
	weights = d2 / (residuals.array().square() + d2).square();
}

void RobustRotationsAveraging::filter_outlier_edges(EpipolarGraph & epi_graph, const std::vector<Eigen::Matrix3d>& Rij)
{
	EpipolarGraph::edge_iterator e_begin, e_end;
	tie(e_begin, e_end) = edges(epi_graph);
	Matrix3d Ri, Rj, R_err;
	Vector3d r_err;
	int edge_idx = 0;

	//vector<double> err_ij(Rij.size());

	double max_difference_radius = deg2rad(m_options.max_difference_degree);
	for (auto it_e = e_begin; it_e != e_end; ++it_e, ++edge_idx)
	{
		size_t i = it_e->m_source;
		size_t j = it_e->m_target;
		if (i > j)	//若 i > j 则交换
		{
			i = i^j;
			j = i^j;
			i = i^j;
		}

		angle_axis_to_rotation_matrix(epi_graph[i].rt.topRows(3), Ri);
		angle_axis_to_rotation_matrix(epi_graph[j].rt.topRows(3), Rj);

		R_err = Rj.transpose() * Rij[edge_idx] * Ri;
		rotation_matrix_to_angle_axis(R_err, r_err);
		double err = r_err.norm();
		//cout << "err = " << err << endl;
		//err_ij[edge_idx] = err;
		if (err > max_difference_radius)
		{
			//cout << "Remove edge: " << i << " - " << j << endl;
			epi_graph[*it_e].is_inlier = false;
		}
	}

	//vector<double> values(err_ij);
	//auto mid = values.begin() + values.size() / 2;
	//std::nth_element(values.begin(), mid, values.end());
	//double median = *mid;
	//// threshold = 5.2 * MEDIAN(ABS(values-median));
	//for (size_t i = 0; i<values.size(); ++i)
	//	values[i] = std::abs(err_ij[i] - median);
	//std::nth_element(values.begin(), mid, values.end());
	//double threshold = median + 5.2*(*mid);
	//cout << "Threshold: " << threshold << endl;

	//tie(e_begin, e_end) = edges(epi_graph);
	//edge_idx = 0;
	//for (auto it_e = e_begin; it_e != e_end; ++it_e, ++edge_idx)
	//{
	//	if (err_ij[edge_idx] > threshold)
	//	{
	//		cout << "Remove Edge: " << it_e->m_source << " - " << it_e->m_target << endl;
	//		epi_graph[*it_e].is_inlier = false;
	//	}
	//}
}
