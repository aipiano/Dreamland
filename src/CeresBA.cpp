#include "CeresBA.h"
#include <map>
#include <thread>

using namespace std;
using namespace Eigen;

CeresBA::CeresBA(CeresBA::Options & options)
{
	m_ba_options = options;
	// Default configuration use a DENSE representation
	m_linear_solver_type = ceres::DENSE_SCHUR;
	m_preconditioner_type = ceres::JACOBI;
	// If Sparse linear solver are available
	// Descending priority order by efficiency (SUITE_SPARSE > CX_SPARSE > EIGEN_SPARSE)
	if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE))
	{
		m_sparse_library_type = ceres::SUITE_SPARSE;
		m_linear_solver_type = ceres::SPARSE_SCHUR;
	}
	else if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CX_SPARSE))
	{
		m_sparse_library_type = ceres::CX_SPARSE;
		m_linear_solver_type = ceres::SPARSE_SCHUR;
	}
	else if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))
	{
		m_sparse_library_type = ceres::EIGEN_SPARSE;
		m_linear_solver_type = ceres::SPARSE_SCHUR;
	}
}

bool CeresBA::solve(Scene & scene, int adjust_type)
{
	ceres::Problem problem;

	//add extrinsics block
	for (size_t i = 0; i < scene.views.size(); ++i)
	{
		auto& view = scene.views[i];
		problem.AddParameterBlock(view.rt.data(), 6);

		vector<int> constant_extrinsics;
		if (!(adjust_type & ADJUST_ROTATIONS))
		{
			constant_extrinsics.push_back(0);
			constant_extrinsics.push_back(1);
			constant_extrinsics.push_back(2);
		}
		if (!(adjust_type & ADJUST_TRANSLATIONS))
		{
			constant_extrinsics.push_back(3);
			constant_extrinsics.push_back(4);
			constant_extrinsics.push_back(5);
		}
		if (!constant_extrinsics.empty())
		{
			if (constant_extrinsics.size() == 6)
			{
				// set the whole parameter block as constant for best performance
				problem.SetParameterBlockConstant(view.rt.data());
			}
			else
			{
				ceres::SubsetParameterization* subset_parameters = new ceres::SubsetParameterization(6, constant_extrinsics);
				problem.SetParameterization(view.rt.data(), subset_parameters);
			}
		}
	}
	// Fix the first extrinsic
	problem.SetParameterBlockConstant(scene.views[0].rt.data());

	//add intrinsics data
	ViewCameraBinder::CameraIterator cam_begin, cam_end;
	tie(cam_begin, cam_end) = scene.view_camera_binder.get_all_cameras();
	for (auto cam_it = cam_begin; cam_it != cam_end; ++cam_it)
	{
		CameraIntrinsic& intrinsic = cam_it->get_intrinsic();
		problem.AddParameterBlock(intrinsic.data(), intrinsic.rows());
		if (!(adjust_type & ADJUST_INTRINSICS))
		{
			problem.SetParameterBlockConstant(intrinsic.data());
		}
	}

	ceres::LossFunction* loss_function = new ceres::HuberLoss(std::sqrt(m_ba_options.huber_loss_width));

	//add tracks
	auto& view_idx_by_view_id = scene.view_idx_by_view_id;
	int num_inlier_tracks = 0;
	for (auto& track : scene.tracks)
	{
		if (!track.is_inlier)
			continue;

		int num_inlier_nodes = 0;
		for (auto& node : track.nodes)
		{
			if (!node.is_inlier)
				continue;

			++num_inlier_nodes;
			std::shared_ptr<Camera> camera = scene.view_camera_binder.get_camera(node.view_id);
			GlobalTransform& view = scene.views[view_idx_by_view_id[node.view_id]];
			ceres::CostFunction* cost_function = camera->create_cost_function(node.observation);

			if (cost_function == nullptr) continue;
			problem.AddResidualBlock(
				cost_function,
				loss_function,
				camera->get_intrinsic().data(),	// Intrinsic
				view.rt.data(),					// View Rotation and Translation
				track.point3d.data()			// Point in 3D space
			);
		}
		if (num_inlier_nodes > 0)
			++num_inlier_tracks;

		if (!(adjust_type & ADJUST_STRUCTURE) && num_inlier_nodes > 0)
		{
			problem.SetParameterBlockConstant(track.point3d.data());
		}
	}

	//TODO: support control points

	// Configure a BA engine and run it
	// Make Ceres automatically detect the bundle structure.
	ceres::Solver::Options ceres_config_options;
	ceres_config_options.preconditioner_type = m_preconditioner_type;
	ceres_config_options.linear_solver_type = m_linear_solver_type;
	ceres_config_options.sparse_linear_algebra_library_type = m_sparse_library_type;
	ceres_config_options.minimizer_progress_to_stdout = false;
	ceres_config_options.logging_type = ceres::SILENT;
	ceres_config_options.num_threads = m_ba_options.multithreaded ? max(std::thread::hardware_concurrency(), (unsigned int)1) : 1;
	ceres_config_options.num_linear_solver_threads = ceres_config_options.num_threads;
	ceres_config_options.function_tolerance = m_ba_options.function_tolerance;
	ceres_config_options.gradient_tolerance = m_ba_options.gradient_tolerance;
	ceres_config_options.parameter_tolerance = m_ba_options.parameter_tolerance;
	ceres_config_options.max_num_iterations = m_ba_options.max_num_iterations;
	ceres_config_options.max_solver_time_in_seconds = m_ba_options.max_solver_time_in_seconds;
	ceres_config_options.use_inner_iterations = m_ba_options.use_inner_iterations;

	// Solve BA
	ceres::Solver::Summary summary;
	ceres::Solve(ceres_config_options, &problem, &summary);

	if (m_ba_options.show_ceres_summary)
		std::cout << summary.FullReport() << std::endl;

	// If error, show failure.
	if (!summary.IsSolutionUsable())
	{
		if (m_ba_options.show_verbose)
			std::cout << "Bundle Adjustment failed." << std::endl;

		m_final_rmse = numeric_limits<double>::max();
		return false;
	}
	else
	{
		m_final_rmse = std::sqrt(summary.final_cost / summary.num_residuals);
		if (m_ba_options.show_verbose)
		{
			// Display statistics about the minimization
			std::cout << std::endl
				<< "Bundle Adjustment statistics (approximated RMSE):\n"
				<< " #views: " << scene.views.size() << "\n"
				<< " #intrinsics: " << scene.view_camera_binder.camera_size() << "\n"
				<< " #tracks: " << scene.tracks.size() << "\n"
				<< " #inlier tracks: " << num_inlier_tracks << "\n"
				<< " #residuals: " << summary.num_residuals << "\n"
				<< " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
				<< " Final RMSE: " << m_final_rmse << "\n"
				<< " Time (s): " << summary.total_time_in_seconds << "\n"
				<< std::endl;
		}
	}

	return true;
}

double CeresBA::final_rmse()
{
	return m_final_rmse;
}
