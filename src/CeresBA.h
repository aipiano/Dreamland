#pragma once

#include <ceres\ceres.h>
#include "Scene.h"
#include "Camera.h"

class CeresBA
{
public:
	enum AdjustType{
		ADJUST_INTRINSICS	= 1,
		ADJUST_ROTATIONS	= 2,
		ADJUST_TRANSLATIONS = 4,
		ADJUST_EXTRINSICS	= ADJUST_ROTATIONS | ADJUST_TRANSLATIONS,
		ADJUST_STRUCTURE	= 8,
		ADJUST_ALL			= ADJUST_INTRINSICS | ADJUST_EXTRINSICS | ADJUST_STRUCTURE,
	};

	struct Options
	{
		bool show_verbose = true;
		bool show_ceres_summary = false;
		bool multithreaded = true;
		bool use_inner_iterations = true;
		int	max_num_iterations = 500;
		double max_solver_time_in_seconds = 3600;
		double function_tolerance = 1e-6;
		double gradient_tolerance = 1e-10;	// typically equals 1e-4 * function_tolerance
		double parameter_tolerance = 1e-8;
		double max_trust_region_radius = 1e12;
		double huber_loss_width = 16.0;
	};

	CeresBA(CeresBA::Options& options);
	bool solve(Scene& scene, int adjust_type = ADJUST_ALL);
	double final_rmse();

private:
	CeresBA::Options m_ba_options;
	ceres::LinearSolverType m_linear_solver_type;
	ceres::PreconditionerType m_preconditioner_type;
	ceres::SparseLinearAlgebraLibraryType m_sparse_library_type;
	double m_final_rmse = DBL_MAX;
};
