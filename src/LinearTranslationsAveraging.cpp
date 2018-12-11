#include "LinearTranslationsAveraging.h"
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
using namespace Eigen;

LinearTranslationsAveraging::LinearTranslationsAveraging(Options options)
{
	m_validator_options.min_num_inliers = options.min_num_inliers;
	m_validator_options.init_max_reproj_error = options.init_max_reproj_error;
	m_validator_options.final_max_reproj_error = options.final_max_reproj_error;
	m_validator_options.refine_with_ba = true;
}

bool LinearTranslationsAveraging::estimate(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, ViewCameraBinder & view_camera_binder, EpipolarGraph & epi_graph)
{
	TripletTranslationsValidator validator(m_validator_options);
	TripletsBuilder triplet_builder;

	Triplets triplets;
	triplet_builder.build(epi_graph);
	triplet_builder.swap(triplets);

	int num_vertex = num_vertices(epi_graph);
	int num_triplets = triplets.size();

	vector<int> cam_triple_count(num_vertex, 0);

#ifdef _DEBUG
	int num_cores = 1;
#else
	int num_cores = omp_get_num_procs();
#endif
	omp_set_num_threads(num_cores);
	vector<CoinBuild> lp_builders(num_cores);

	std::vector<double> scales_ik;
	std::vector<double> scales_jk;
	vector<unsigned char> mask;

	std::shared_ptr<ScalesSelectorKernel> kernel = make_shared<ScalesSelectorKernel>();
	Ransac<double, double> scales_selector(kernel, 1, 0.05, 0.99, 256);

	int triplets_count = triplets.size();
	#pragma omp parallel for private(scales_ik, scales_jk, mask)
	for (int triplet_idx = 0; triplet_idx < triplets_count; ++triplet_idx)
	{
		auto& t = triplets[triplet_idx];
		// must have idx_i < idx_j < idx_k.
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
		PairwiseMatches triplet_pairs;
		TracksBuilder tracks_builder;
		triplet_pairs.reserve(2);
		triplet_pairs.emplace_back(view_i, view_j);
		triplet_pairs.emplace_back(view_i, view_k);
		tracks_builder.build(feature_pool, match_pool, triplet_pairs);
		tracks_builder.filter(3);
		Tracks triplet_tracks;
		tracks_builder.swap(triplet_tracks);

		size_t num_tracks = triplet_tracks.size();
		if (num_tracks < m_validator_options.min_num_inliers)
			continue;

		// transforms correspond to edge_ij, edge_ik and edge_jk.
		auto transform_ij = epi_graph[t.e_ij];
		auto transform_ik = epi_graph[t.e_ik];
		auto transform_jk = epi_graph[t.e_jk];
		auto& rt_ij = transform_ij.rt;
		auto& rt_ik = transform_ik.rt;
		auto& rt_jk = transform_jk.rt;
		if (transform_ij.src_id != view_i)
		{	// change transform direction.
			transform_ij.src_id = view_i; transform_ij.dst_id = view_j;
			rt_ij = reversed_transform(rt_ij);	
		}
		if (transform_ik.src_id != view_i)
		{
			transform_ik.src_id = view_i; transform_ik.dst_id = view_k;
			rt_ik = reversed_transform(rt_ik);
		}
		if (transform_jk.src_id != view_j)
		{
			transform_jk.src_id = view_j; transform_jk.dst_id = view_k;
			rt_jk = reversed_transform(rt_jk);
		}

		// calculate scales of edge_ik and edge_jk. the scale of edge_ij is 1.
		scales_ik.resize(num_tracks);
		scales_jk.resize(num_tracks);
		Quaterniond q_ij, q_ik, q_jk;
		angle_axis_to_quaternion(rt_ij.topRows(3), q_ij);
		angle_axis_to_quaternion(rt_ik.topRows(3), q_ik);
		angle_axis_to_quaternion(rt_jk.topRows(3), q_jk);
		Vector3d t_ij = rt_ij.bottomRows(3);
		Vector3d t_ik = rt_ik.bottomRows(3);
		Vector3d t_jk = rt_jk.bottomRows(3);
		
		size_t track_idx = 0;
		Vector3d xRx;
		for (auto it_track = triplet_tracks.begin(); it_track != triplet_tracks.end(); ++it_track, ++track_idx)
		{
			auto& nodes = it_track->nodes;
			Vector3d x_i, x_j, x_k;
			for (auto& node : nodes)
			{
				if (node.view_id == view_i)
					x_i = camera_i->image_to_world(node.observation);
				else if (node.view_id == view_j)
					x_j = camera_j->image_to_world(node.observation);
				else
					x_k = camera_k->image_to_world(node.observation);
			}
			/*
			- From the pairwise reconstruction with image i, j, we compute its depth d_ij^i in  \ 
			- the image i	while assuming unit baseline length.
			- Similarly, we can calculate d_ik^i which is the depth of X in the image i from the \
			- reconstruction of image i, k. 
			- The scale s_ik is then estimated as d_ij^i / d_ik^i.
			*/
			xRx = x_j.cross(q_ij * x_i);
			double depth_ij = xRx.dot(x_j.cross(t_ij)) / xRx.squaredNorm();
			xRx = x_k.cross(q_ik * x_i);
			double depth_ik = xRx.dot(x_k.cross(t_ik)) / xRx.squaredNorm();
			scales_ik[track_idx] = depth_ij / depth_ik;

			double depth_ji = (depth_ij * (q_ij * x_i) + rt_ij.bottomRows(3))[2];
			xRx = x_k.cross(q_jk * x_j);
			double depth_jk = xRx.dot(x_k.cross(t_jk)) / xRx.squaredNorm();
			scales_jk[track_idx] = depth_ji / depth_jk;
		}

		// estimate final edge scale by RANSAC
		double scale_ik, scale_jk;
		scales_selector.run(scales_ik, scale_ik, mask);
		scale_ik = average_scales(scales_ik, mask);

		scales_selector.run(scales_jk, scale_jk, mask);
		scale_jk = average_scales(scales_jk, mask);

		// get view positions from extrinsic transforms
		Vector3d c_ij = -(q_ij.conjugate() * t_ij);	// scale_ij == 1
		Vector3d c_ik = -(q_ik.conjugate() * t_ik) * scale_ik;
		Vector3d c_jk = -(q_jk.conjugate() * t_jk) * scale_jk;
		c_ik = 0.5 * (c_ik + c_ij + c_jk);
		rt_ik.bottomRows(3) = -(q_ik * c_ik);	// change position to translation

		if (validator.acceptable(transform_ij, transform_ik, camera_i, camera_j, camera_k, triplet_tracks))
		{
			rt_jk = relative_transform_between(rt_ij, rt_ik);
		}
		else
		{
			continue;
		}

		// build linear programing block.
#ifdef _DEBUG
		int thread_number = 0;
#else
		int thread_number = omp_get_thread_num();
#endif
		CoinBuild& lp_builder = lp_builders[thread_number];
		set_lp_builder(
			lp_builder, num_vertex, num_triplets,
			rt_ij, rt_ik, rt_jk,
			idx_i, idx_j, idx_k,
			triplet_idx
		);

		// update progress.
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

double LinearTranslationsAveraging::average_scales(std::vector<double>& scales, vector<unsigned char>& mask)
{
	double sum_scales = 0;
	size_t num_scales = 0;
	size_t total = scales.size();
	for (size_t i = 0; i < total; ++i)
	{
		if (mask[i] == 0)
			continue;
		sum_scales += scales[i];
		++num_scales;
	}
	return sum_scales / num_scales;
}

void LinearTranslationsAveraging::set_lp_builder(
	CoinBuild & lp_builder, int num_vertices, int num_triplets,
	CameraExtrinsic & rt_ij, CameraExtrinsic & rt_ik, CameraExtrinsic & rt_jk, 
	int idx_i, int idx_j, int idx_k, int triplet_idx
)
{
	int cols = 3 * num_vertices + num_triplets + 1;

	Matrix<double, 9, 6, RowMajor> row_data_pos;
	Matrix<double, 9, 6, RowMajor> row_data_nag;
	Matrix3d R_ij, R_ik, R_jk;
	angle_axis_to_rotation_matrix(rt_ij.topRows(3), R_ij);
	angle_axis_to_rotation_matrix(rt_ik.topRows(3), R_ik);
	angle_axis_to_rotation_matrix(rt_jk.topRows(3), R_jk);

	row_data_pos.block(0, 0, 3, 3) = -R_ij;
	row_data_pos.block(3, 0, 3, 3) = -R_ik;
	row_data_pos.block(6, 0, 3, 3) = -R_jk;
	row_data_pos.col(3).setOnes();

	row_data_pos.block(0, 4, 3, 1) = -rt_ij.bottomRows(3);
	row_data_pos.block(3, 4, 3, 1) = -rt_ik.bottomRows(3);
	row_data_pos.block(6, 4, 3, 1) = -rt_jk.bottomRows(3);
	row_data_pos.col(5).setConstant(-1);

	row_data_nag = -row_data_pos;
	row_data_nag.col(5).setConstant(-1);

	std::array<int, 6> indices{
		idx_i * 3, idx_i * 3 + 1, idx_i * 3 + 2,
		idx_j * 3,
		num_vertices * 3 + triplet_idx, cols - 1
	};
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
}

bool LinearTranslationsAveraging::solve_linear_system(EpipolarGraph & epi_graph, std::vector<CoinBuild>& lp_builders)
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

void ScalesSelectorKernel::run_kernel(const std::vector<double>& samples, std::vector<double>& models)
{
	double sum_scales = 0;
	size_t num_samples = samples.size();
	models.clear();
	
	if (num_samples == 0) return;

	for (size_t i = 0; i < num_samples; ++i)
	{
		sum_scales += samples[i];
	}
	models.push_back(sum_scales / num_samples);
}

void ScalesSelectorKernel::compute_error(const std::vector<double>& samples, const double & model, Eigen::VectorXd & errors)
{
	size_t num_samples = samples.size();
	for (size_t i = 0; i < num_samples; ++i)
	{
		errors[i] = abs(samples[i] - model);
	}
}
