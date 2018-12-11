#include "TrilinearTranslationsAveraging.h"
#include "TracksBuilder.h"
#include "Utils.h"
#include "Ransac.hpp"
#include "TripletsBuilder.h"
#include "TripletTranslationsValidator.h"
#include <Eigen/Eigen>
#include <ClpSimplex.hpp>
#include <CoinBuild.hpp>
#include <CoinPackedVector.hpp>
#include <omp.h>

using namespace std;
using namespace boost;
using namespace Eigen;

TrilinearTranslationsAveraging::TrilinearTranslationsAveraging(Options options)
{
	m_options = options;

	m_validator_options.min_num_inliers = options.min_num_inliers;
	m_validator_options.init_max_reproj_error = options.init_max_reproj_error;
	m_validator_options.final_max_reproj_error = options.final_max_reproj_error;
	m_validator_options.refine_with_ba = options.refine_with_ba;
}

bool TrilinearTranslationsAveraging::estimate(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, ViewCameraBinder & view_camera_binder, EpipolarGraph & epi_graph)
{
	TripletTranslationsValidator validator(m_validator_options);
	TripletsBuilder triplet_builder;
	
	Triplets triplets;
	triplet_builder.build(epi_graph);
	triplet_builder.swap(triplets);

	size_t num_vertex = num_vertices(epi_graph);
	size_t cols = 3 * num_vertex + triplets.size() + 1;
	vector<int> cam_triple_count(num_vertex, 0);

#ifdef _DEBUG
	int num_cores = 1;
	vector<CoinBuild> lp_builders(num_cores);
	omp_set_num_threads(num_cores);
#else
	int num_cores = omp_get_num_procs();
	vector<CoinBuild> lp_builders(num_cores);
	omp_set_num_threads(num_cores);
#endif

	int triplets_count = triplets.size();
	cout << "# Triplets: " << triplets_count << endl;

	// each thread hold a group of instances of below objects.
	TracksBuilder tracks_builder;
	vector<Matrix<double, 9, 1>> triple_points;
	vector<unsigned char> mask;
	Matrix<double, 9, 6, RowMajor> row_data_pos;
	Matrix<double, 9, 6, RowMajor> row_data_nag;

	#pragma omp parallel for schedule(dynamic) private(tracks_builder, triple_points, mask, row_data_pos, row_data_nag)
	for (int triplet_idx = 0; triplet_idx < triplets_count; ++triplet_idx)
	{
		auto& t = triplets[triplet_idx];
		// must have idx_i < idx_j < idx_k;
		int idx_i = t.e_ij.m_source;
		int idx_j = t.e_ij.m_target;
		int idx_k = t.e_ik.m_target;
		// view ids correspond to graph nodes.
		auto view_i = epi_graph[idx_i].view_id;
		auto view_j = epi_graph[idx_j].view_id;
		auto view_k = epi_graph[idx_k].view_id;
		// camera models correspond to views.
		auto camera_i = view_camera_binder.get_camera(view_i);
		auto camera_j = view_camera_binder.get_camera(view_j);
		auto camera_k = view_camera_binder.get_camera(view_k);

		// build triplet tracks
		PairwiseMatches triple_pairs;
		triple_pairs.reserve(2);
		triple_pairs.emplace_back(view_i, view_j);
		triple_pairs.emplace_back(view_i, view_k);

		tracks_builder.build(feature_pool, match_pool, triple_pairs);
		tracks_builder.filter(3);
		Tracks triple_tracks;
		tracks_builder.swap(triple_tracks);

		size_t num_tracks = triple_tracks.size();
		if (num_tracks < 4 || num_tracks < m_options.min_num_tracks)
			continue;

		// transforms correspond to edge_ij, edge_ik and edge_jk.
		// eij的两个端点就是view_i和view_j，不过变换方向可能相反
		auto transform_ij = epi_graph[t.e_ij];
		auto transform_ik = epi_graph[t.e_ik];
		auto transform_jk = epi_graph[t.e_jk];
		//if (transform_ij.src_id != view_i)
		//{	// change transform direction.
		//	transform_ij.src_id = view_i; transform_ij.dst_id = view_j;
		//	transform_ij.rt = reversed_transform(transform_ij.rt);
		//}
		//if (transform_ik.src_id != view_i)
		//{
		//	transform_ik.src_id = view_i; transform_ik.dst_id = view_k;
		//	transform_ik.rt = reversed_transform(transform_ik.rt);
		//}
		//if (transform_jk.src_id != view_j)
		//{
		//	transform_jk.src_id = view_j; transform_jk.dst_id = view_k;
		//	transform_jk.rt = reversed_transform(transform_jk.rt);
		//}
		Matrix3d R_ij, R_ik, R_jk;
		angle_axis_to_rotation_matrix(transform_ij.rt.topRows(3), R_ij);
		angle_axis_to_rotation_matrix(transform_ik.rt.topRows(3), R_ik);
		angle_axis_to_rotation_matrix(transform_jk.rt.topRows(3), R_jk);

		// prepair data for RANSAC
		triple_points.resize(num_tracks);
		size_t track_idx = 0;
		for (auto it_track = triple_tracks.begin(); it_track != triple_tracks.end(); ++it_track, ++track_idx)
		{
			auto& nodes = it_track->nodes;
			for (int n = 0; n < 3; ++n)
			{
				auto& node = nodes[n];
				if (node.view_id == view_i)
					triple_points[track_idx].topRows(3) = camera_i->image_to_world(node.observation);
				else if (node.view_id == view_j)
					triple_points[track_idx].middleRows(3, 3) = camera_i->image_to_world(node.observation);
				else
					triple_points[track_idx].bottomRows(3) = camera_k->image_to_world(node.observation);
			}
		}
		//Do RANSAC
		std::shared_ptr<TrilinearKernel> kernel = make_shared<TrilinearKernel>();
		kernel->set_triplet_rotations(R_ij, R_ik, R_jk);

		double focal = camera_i->get_intrinsic()[0] + camera_j->get_intrinsic()[0] + camera_k->get_intrinsic()[0];	//TODO: just for test
		focal /= 3.0;
		Ransac<Eigen::Matrix<double, 9, 1>, Eigen::Matrix<double, 9, 1>> trilinear_refiner(kernel, 4, 3. / focal, 0.99, 256);

		Matrix<double, 9, 1> model;
		if (!trilinear_refiner.run(triple_points, model, mask))
			continue;

		int num_inliers = cv::countNonZero(mask);
		if (num_inliers < m_validator_options.min_num_inliers)
			continue;

		// 保证优化后的translation与原来的方向相同
		Vector3d T12 = model.topRows(3);
		if (transform_ij.rt.bottomRows(3).dot(T12) < 0)
			model *= -1;

		transform_ij.rt.bottomRows(3) = model.topRows(3);
		transform_ik.rt.bottomRows(3) = model.middleRows(3, 3);
		// validator will detect outliers automatically.
		if (validator.acceptable(transform_ij, transform_ik, camera_i, camera_j, camera_k, triple_tracks))
		{
			// transform_ij and transform_ik have been updated. update transform_jk now.
			transform_jk.rt = relative_transform_between(transform_ij.rt, transform_ik.rt);
			angle_axis_to_rotation_matrix(transform_jk.rt.topRows(3), R_jk);
		}
		else
		{
			continue;
		}
		
		//Build Linear Programing
#ifdef _DEBUG
		int thread_number = 0;
#else
		int thread_number = omp_get_thread_num();
#endif
		row_data_pos.block(0, 0, 3, 3) = -R_ij;
		row_data_pos.block(3, 0, 3, 3) = -R_ik;
		row_data_pos.block(6, 0, 3, 3) = -R_jk;
		row_data_pos.col(3).setOnes();

		row_data_pos.block(0, 4, 3, 1) = -transform_ij.rt.bottomRows(3);
		row_data_pos.block(3, 4, 3, 1) = -transform_ik.rt.bottomRows(3);
		row_data_pos.block(6, 4, 3, 1) = -transform_jk.rt.bottomRows(3);
		row_data_pos.col(5).setConstant(-1);

		row_data_nag = -row_data_pos;
		row_data_nag.col(5).setConstant(-1);

		std::array<int, 6> indices{
			idx_i * 3, idx_i * 3 + 1, idx_i * 3 + 2,
			idx_j * 3,
			(int)(num_vertex * 3 + triplet_idx), (int)(cols - 1)
		};

		CoinBuild& lp_builder = lp_builders[thread_number];
		//R12
		for (int r = 0; r < 3; ++r)
		{
			lp_builder.addRow(indices.size(), indices.data(), row_data_pos.row(r).data(), -COIN_DBL_MAX, 0);
			lp_builder.addRow(indices.size(), indices.data(), row_data_nag.row(r).data(), -COIN_DBL_MAX, 0);
			indices[3]++;
		}
		//R13
		indices[3] = idx_k * 3;
		for (int r = 3; r < 6; ++r)
		{
			lp_builder.addRow(indices.size(), indices.data(), row_data_pos.row(r).data(), -COIN_DBL_MAX, 0);
			lp_builder.addRow(indices.size(), indices.data(), row_data_nag.row(r).data(), -COIN_DBL_MAX, 0);
			indices[3]++;
		}
		//R23
		indices[0] = idx_j * 3;
		indices[1] = idx_j * 3 + 1;
		indices[2] = idx_j * 3 + 2;
		indices[3] = idx_k * 3;
		for (int r = 6; r < 9; ++r)
		{
			lp_builder.addRow(indices.size(), indices.data(), row_data_pos.row(r).data(), -COIN_DBL_MAX, 0);
			lp_builder.addRow(indices.size(), indices.data(), row_data_nag.row(r).data(), -COIN_DBL_MAX, 0);
			indices[3]++;
		}

		// update progress
		#pragma omp critical
		{
			++cam_triple_count[idx_i];
			++cam_triple_count[idx_j];
			++cam_triple_count[idx_k];
			cout << "Triplet Match: " << idx_i << ", " << idx_j << ", " << idx_k << endl;
		}
	}
	
	// 判断是否有不属于任何triplet的孤立节点，有则删除
	for (size_t i = 0; i < num_vertex; ++i)
	{
		if (cam_triple_count[i] <= 0)
		{
			epi_graph[i].is_inlier = false;
			//clear_vertex(i, epi_graph);
			cout << "Remove Vertex: " << i << endl;
		}
		else
			epi_graph[i].is_inlier = true;
	}

	return solve_linear_system(epi_graph, lp_builders);
}

bool TrilinearTranslationsAveraging::solve_linear_system(EpipolarGraph & epi_graph, vector<CoinBuild> & lp_builders)
{
	size_t num_vertex = num_vertices(epi_graph);
	int variable_count = lp_builders[0].numberColumns();

	ClpSimplex lp_solver;	// simplex method is much faster than interior method for COIN implementation.
	lp_solver.resize(0, variable_count);
	lp_solver.setObjCoeff(variable_count - 1, 1);

	// find the first inlier view.
	int t0 = 0;
	for (size_t i = 0; i < num_vertex; ++i)
	{
		t0 = i;
		if (epi_graph[i].is_inlier)
			break;
	}
	if (t0 == num_vertex) return false;	// all views are outlier.
	// T_0 = (0, 0, 0). Fix all outliers and the first view.
	for (int i = 0; i < t0 * 3 + 3; ++i)
	{
		lp_solver.setColLower(i, 0);
		lp_solver.setColUpper(i, 0);
	}
	for (int i = t0 * 3 + 3; i < 3 * num_vertex; ++i)
	{
		if (epi_graph[i / 3].is_inlier)
		{
			lp_solver.setColLower(i, -COIN_DBL_MAX);
			lp_solver.setColUpper(i, COIN_DBL_MAX);
		}
		else // fix outliers
		{
			lp_solver.setColLower(i, 0);
			lp_solver.setColUpper(i, 0);
		}
	}
	//lambda >= 1;
	for (int i = 3 * num_vertex; i < variable_count - 1; ++i)
	{
		lp_solver.setColLower(i, 1);
		lp_solver.setColUpper(i, COIN_DBL_MAX);
	}
	//gamma >= 0;
	lp_solver.setColLower(variable_count - 1, 0);
	lp_solver.setColUpper(variable_count - 1, COIN_DBL_MAX);

	for (auto& lp_builder : lp_builders)
	{
		lp_solver.addRows(lp_builder);
	}
	lp_solver.setLogLevel(0);
	lp_solver.dual();

	double error = -1;
	if (lp_solver.status() == 0)
	{
		VectorXd Ts = Map<VectorXd>(const_cast<double*>(lp_solver.getColSolution()), 3 * num_vertex);
		error = lp_solver.getObjValue();
		cout << "Final objective value (gamma) = " << error << endl;

		//将结果保存到每个相机中
		for (size_t i = 0; i < num_vertex; ++i)
		{
			if (epi_graph[i].is_inlier)
				epi_graph[i].rt.bottomRows(3) = Ts.middleRows(i * 3, 3);
		}

		return true;
	}
	else
	{
		return false;
	}
}

////////////////////////////////////////////////////////////////
//				TrilinearKernel
////////////////////////////////////////////////////////////////

//int TrilinearKernel::run_kernel(const Eigen::Ref<const Eigen::MatrixXd>& m1, const Eigen::Ref<const Eigen::MatrixXd>& m2, Eigen::MatrixXd& models)
//{
//	if (m1.cols() < 3) return 0;
//
//	typedef Matrix<double, 2, 3> Matrix23d;
//	MatrixXd A = MatrixXd::Zero(12 * m1.cols(), 9);
//	Vector3d x1, x2, x3;
//	Matrix3d hx1, hx2, hx3;
//	Vector2d xRx31, xRx21, xRx32, xRx12, xRx23, xRx13;
//	Matrix23d xR12, xR13, xR23, phx2, phx3;
//
//	for (int i = 0; i < m1.cols(); ++i)
//	{
//		//m1中保存点在前两个图像中的像，m2保存在第三个图像中的像
//		x1 = m1.block(0, i, 3, 1);
//		x2 = m1.block(3, i, 3, 1);
//		x3 = m2.block(0, i, 3, 1);
//
//		hx1 = cross_mat(x1);
//		hx2 = cross_mat(x2);
//		hx3 = cross_mat(x3);
//
//		xRx31 = (hx3*m_R13*x1).topRows(2);
//		xRx21 = (hx2*m_R12*x1).topRows(2);
//
//		xRx32 = (hx3*m_R23*x2).topRows(2);
//		xRx12 = (hx1*m_R12.transpose()*x2).topRows(2);
//
//		xRx23 = (hx2*m_R23.transpose()*x3).topRows(2);
//		xRx13 = (hx1*m_R13.transpose()*x3).topRows(2);
//
//		xR12 = (hx1*m_R12.transpose()).topRows(2);
//		xR13 = (hx1*m_R13.transpose()).topRows(2);
//		xR23 = (hx2*m_R23.transpose()).topRows(2);
//
//		phx2 = hx2.topRows(2);
//		phx3 = hx3.topRows(2);
//
//		size_t start_row = i * 12;
//		kron(-xRx31, phx2, A.block(start_row, 0, 4, 3));
//		kron(phx3, xRx21, A.block(start_row, 3, 4, 3));
//		kron(-xR12, xRx32, A.block(start_row + 4, 0, 4, 3));
//		kron(-xRx12, phx3, A.block(start_row + 4, 6, 4, 3));
//		kron(xRx23, xR13, A.block(start_row + 8, 3, 4, 3));
//		kron(-xR23, xRx13, A.block(start_row + 8, 6, 4, 3));
//	}
//
//	Matrix<double, 9, 1> T;
//	JacobiSVD<Matrix<double, -1, 9>> svd(A, ComputeFullV);
//	T = svd.matrixV().rightCols(1);
//	models = T / T.topRows(3).norm();
//
//	return 1;
//}
//
//void TrilinearKernel::compute_error(const Eigen::Ref<const Eigen::MatrixXd>& m1, const Eigen::Ref<const Eigen::MatrixXd>& m2, const Eigen::Ref<const Eigen::MatrixXd>& model, Eigen::VectorXd& errors)
//{
//	size_t n = m1.cols();
//	errors.resize(n);
//
//	Vector3d T12 = model.topRows(3);
//	Vector3d T13 = model.middleRows(3, 3);
//
//	for (size_t i = 0; i < n; ++i)
//	{
//		Vector3d x1 = m1.block(0, i, 3, 1);
//		Vector3d x2 = m1.block(3, i, 3, 1);
//		Vector3d x3 = m2.block(0, i, 3, 1);
//
//		Matrix3d xm = (T12*x1.transpose()*m_R13.transpose() - m_R12*x1*T13.transpose()) * cross_mat(x3);
//		Vector3d test_x2 = xm.col(0) + xm.col(1) + xm.col(2);
//
//		//TODO: 用余弦距离代替
//		double z = test_x2[2];
//		if (abs(z) < 10.0*DBL_EPSILON)
//		{
//			errors[i] = DBL_MAX;
//		}
//		test_x2 /= z;
//		errors[i] = (x2 - test_x2).norm();
//	}
//
//	vector<double> vec_errors(errors.data(), errors.data() + errors.rows());
//	int ns = vec_errors.size();
//}
//
//void TrilinearKernel::set_triplet_rotations(
//	const Eigen::Ref<const Eigen::Matrix3d>& R12,
//	const Eigen::Ref<const Eigen::Matrix3d>& R13,
//	const Eigen::Ref<const Eigen::Matrix3d>& R23
//)
//{
//	m_R12 = R12;
//	m_R13 = R13;
//	m_R23 = R23;
//}

// 用SVD方法求解三线性约束，不使用凸优化方法是为了兼容全景相机
void TrilinearKernel::run_kernel(
	const std::vector<Eigen::Matrix<double, 9, 1>>& samples,
	std::vector<Eigen::Matrix<double, 9, 1>>& models
)
{
	models.clear();
	if (samples.size() < 4) return;

	Vector3d x1, x2, x3;
	Matrix3d hx1, hx2, hx3;
	Vector3d xRx31, xRx21, xRx32, xRx12, xRx23, xRx13;
	Matrix3d xR12, xR13, xR23, phx2, phx3;

	m_mat99.setZero();

	for (int i = 0; i < samples.size(); ++i)
	{
		//m1中保存点在前两个图像中的像，m2保存在第三个图像中的像
		x1 = samples[i].topRows(3);
		x2 = samples[i].middleRows(3, 3);
		x3 = samples[i].bottomRows(3);

		hx1 = cross_mat(x1);
		hx2 = cross_mat(x2);
		hx3 = cross_mat(x3);

		xRx31 = x3.cross(m_q13*x1);
		xRx21 = x2.cross(m_q12*x1);

		xRx32 = x3.cross(m_q23*x2);
		xRx12 = x1.cross(m_q12i*x2);

		xRx23 = x2.cross(m_q23i*x3);
		xRx13 = x1.cross(m_q13i*x3);

		xR12.noalias() = hx1*m_R12.transpose();
		xR13.noalias() = hx1*m_R13.transpose();
		xR23.noalias() = hx2*m_R23.transpose();
		
		// AtA + CtC
		m_mat99.block<3, 3>(0, 0).triangularView<Lower>() += xRx31.squaredNorm() * hx2.transpose() * hx2;
		m_mat99.block<3, 3>(0, 0).triangularView<Lower>() += xRx32.squaredNorm() * xR12.transpose() * xR12;
		// BtB + EtE
		m_mat99.block<3, 3>(3, 3).triangularView<Lower>() += xRx21.squaredNorm() * hx3.transpose() * hx3;
		m_mat99.block<3, 3>(3, 3).triangularView<Lower>() += xRx23.squaredNorm() * xR13.transpose() * xR13;
		// DtD + EtE
		m_mat99.block<3, 3>(6, 6).triangularView<Lower>() += xRx12.squaredNorm() * hx3.transpose() * hx3;
		m_mat99.block<3, 3>(6, 6).triangularView<Lower>() += xRx13.squaredNorm() * xR23.transpose() * xR23;
		// BtA
		m_mat99.block<3, 3>(3, 0) += -hx3.transpose() * xRx31 * xRx21.transpose() * hx2;
		// DtC
		m_mat99.block<3, 3>(6, 0) += hx3.transpose() * xRx32 * xRx12.transpose() * xR12;
		// FtE
		m_mat99.block<3, 3>(6, 3) += -xR23.transpose() * xRx23 * xRx13.transpose() * xR13;
	}

	// Only the lower triangular part of the input matrix is referenced.
	m_eig_solver.compute(m_mat99);
	models.resize(1);
	models[0] = m_eig_solver.eigenvectors().col(0);
	models[0] /= models[0].topRows(3).norm();
}

//void TrilinearKernel::run_kernel(
//	const std::vector<Eigen::Matrix<double, 9, 1>>& samples,
//	std::vector<Eigen::Matrix<double, 9, 1>>& models
//)
//{
//	models.clear();
//	if (samples.size() < 4) return;
//	
//	typedef Matrix<double, 2, 3> Matrix23d;
//	MatrixXd A = MatrixXd::Zero(12 * samples.size(), 9);
//	Vector3d x1, x2, x3;
//	Matrix3d hx1, hx2, hx3;
//	Vector2d xRx31, xRx21, xRx32, xRx12, xRx23, xRx13;
//	Matrix23d xR12, xR13, xR23, phx2, phx3;
//	
//	for (int i = 0; i < samples.size(); ++i)
//	{
//		//m1中保存点在前两个图像中的像，m2保存在第三个图像中的像
//		x1 = samples[i].topRows(3);
//		x2 = samples[i].middleRows(3, 3);
//		x3 = samples[i].bottomRows(3);
//	
//		hx1 = cross_mat(x1);
//		hx2 = cross_mat(x2);
//		hx3 = cross_mat(x3);
//	
//		xRx31 = x3.cross(m_q13*x1).topRows(2);
//		xRx21 = x2.cross(m_q12*x1).topRows(2);
//	
//		xRx32 = x3.cross(m_q23*x2).topRows(2);
//		xRx12 = x1.cross(m_q12i*x2).topRows(2);
//	
//		xRx23 = x2.cross(m_q23i*x3).topRows(2);
//		xRx13 = x1.cross(m_q13i*x3).topRows(2);
//	
//		xR12 = (hx1*m_R12.transpose()).topRows(2);
//		xR13 = (hx1*m_R13.transpose()).topRows(2);
//		xR23 = (hx2*m_R23.transpose()).topRows(2);
//	
//		phx2 = hx2.topRows(2);
//		phx3 = hx3.topRows(2);
//		
//		size_t start_row = i * 12;
//		kron(-xRx31, phx2, A.block(start_row, 0, 4, 3));
//		kron(phx3, xRx21, A.block(start_row, 3, 4, 3));
//		kron(-xR12, xRx32, A.block(start_row + 4, 0, 4, 3));
//		kron(-xRx12, phx3, A.block(start_row + 4, 6, 4, 3));
//		kron(xRx23, xR13, A.block(start_row + 8, 3, 4, 3));
//		kron(-xR23, xRx13, A.block(start_row + 8, 6, 4, 3));
//	}
//	
//	Matrix<double, 9, 1> T;
//	JacobiSVD<Matrix<double, -1, 9>> svd(A, ComputeFullV);
//	T = svd.matrixV().rightCols(1);
//	models.resize(1);
//	models[0] = T / T.topRows(3).norm();
//}

void TrilinearKernel::compute_error(
	const std::vector<Eigen::Matrix<double, 9, 1>>& samples,
	const Eigen::Matrix<double, 9, 1> & model,
	Eigen::VectorXd & errors
)
{
	size_t n = samples.size();
	errors.resize(n);

	Vector3d T12 = model.topRows(3);
	Vector3d T13 = model.middleRows(3, 3);
	Vector3d T23 = model.bottomRows(3);
	Vector3d x1, x2, x3;

	Matrix3d xm;
	Vector3d tx1, tx2, tx3;

	for (size_t i = 0; i < n; ++i)
	{
		x1 = samples[i].topRows(3);
		x2 = samples[i].middleRows(3, 3);
		x3 = samples[i].bottomRows(3);

		xm = (m_q13i*x3*((m_q23i*T23).transpose()) - m_q13i*T13*((m_q23i*x3).transpose())) * cross_mat(x2);
		tx1 = xm.col(0) + xm.col(1) + xm.col(2);

		xm = (T12*((m_q13*x1).transpose()) - m_q12*x1*T13.transpose()) * cross_mat(x3);
		tx2 = xm.col(0) + xm.col(1) + xm.col(2);

		xm = (T23*((m_q12i*x2).transpose()) + m_q23*x2*((m_q12i*T12).transpose())) * cross_mat(x1);
		tx3 = xm.col(0) + xm.col(1) + xm.col(2);

		if (abs(tx1[2]) < 10.0*DBL_EPSILON || 
			abs(tx2[2]) < 10.0*DBL_EPSILON ||
			abs(tx3[2]) < 10.0*DBL_EPSILON)
		{
			errors[i] = DBL_MAX;
			continue;
		}

		//TODO: 改为其他度量方式，以支持全景图像
		tx1 /= tx1[2];
		tx2 /= tx2[2];
		tx3 /= tx3[2];
		errors[i] = max((x1 - tx1).norm(), max((x2 - tx2).norm(), (x3 - tx3).norm()));
	}
}

void TrilinearKernel::set_triplet_rotations(
	const Eigen::Ref<const Eigen::Matrix3d>& R12, 
	const Eigen::Ref<const Eigen::Matrix3d>& R13, 
	const Eigen::Ref<const Eigen::Matrix3d>& R23
)
{
	m_R12 = R12;
	m_R13 = R13;
	m_R23 = R23;

	m_q12 = Quaterniond(R12);
	m_q13 = Quaterniond(R13);
	m_q23 = Quaterniond(R23);

	m_q12i = m_q12.conjugate();
	m_q13i = m_q13.conjugate();
	m_q23i = m_q23.conjugate();
}

//void TrilinearKernel::set_triplet_cameras(const std::shared_ptr<Camera> camera1, const std::shared_ptr<Camera> camera2, const std::shared_ptr<Camera> camera3)
//{
//	m_camera1 = camera1;
//	m_camera2 = camera2;
//	m_camera3 = camera3;
//}
