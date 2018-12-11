#pragma once

//------------------
//-- Bibliography --
//------------------
//- [1] "A Global Linear Method for Camera Pose Registration"
//- Authors: Nianjuan Jiang, Zhaopeng Cui, Ping Tan
//- Date: 2013.
//- Conference: ICCV.
//

#include "GlobalTranslationsEstimator.h"
#include "TripletMatch.h"
#include "RansacKernel.hpp"
#include "TripletTranslationsValidator.h"
#include <map>
#include <vector>
#include <list>
#include <CoinBuild.hpp>

class ScalesSelectorKernel final : public RansacKernel<double, double>
{
public:
	// 通过 RansacKernel 继承
	virtual void run_kernel(const std::vector<double>& samples, std::vector<double>& models) override;
	virtual void compute_error(const std::vector<double>& samples, const double & model, Eigen::VectorXd & errors) override;
};

class LinearTranslationsAveraging final : public GlobalTranslationsEstimator
{
public:
	struct Options
	{
		size_t min_num_inliers = 30;			// min inliers satisfy the init_max_reproj_error
		double init_max_reproj_error = 10.0;	// max allowed reproject error after triangulate
		double final_max_reproj_error = 2.0;	// max allowed reproject error after refinement
	};

	LinearTranslationsAveraging(Options options);

	// 通过 GlobalTranslationsEstimator 继承
	virtual bool estimate(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, ViewCameraBinder & view_camera_binder, EpipolarGraph & epi_graph) override;

private:
	bool solve_linear_system(EpipolarGraph& epi_graph, std::vector<CoinBuild>& lp_builders);
	double average_scales(std::vector<double>& scales, std::vector<unsigned char>& mask);
	void set_lp_builder(
		CoinBuild& lp_builder,
		int num_vertices,
		int num_triplets,
		CameraExtrinsic& rt_ij,
		CameraExtrinsic& rt_ik,
		CameraExtrinsic& rt_jk,
		int idx_i, int idx_j, int idx_k,
		int triplet_idx
	);

private:
	TripletTranslationsValidator::Options m_validator_options;
};
