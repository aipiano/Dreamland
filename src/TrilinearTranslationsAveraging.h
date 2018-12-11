#pragma once

#include "GlobalTranslationsEstimator.h"
#include "TripletMatch.h"
//#include "SampleConsensusKernel.h"
#include "RansacKernel.hpp"
#include "TripletTranslationsValidator.h"
#include <map>
#include <vector>
#include <list>
#include <CoinBuild.hpp>

//class TrilinearKernel final : public SampleConsensusKernel
//{
//public:
//	// 通过 SampleConsensusKernel 继承
//	virtual int run_kernel(const Eigen::Ref<const Eigen::MatrixXd>& m1, const Eigen::Ref<const Eigen::MatrixXd>& m2, Eigen::MatrixXd& models) override;
//	virtual void compute_error(const Eigen::Ref<const Eigen::MatrixXd>& m1, const Eigen::Ref<const Eigen::MatrixXd>& m2, const Eigen::Ref<const Eigen::MatrixXd>& model, Eigen::VectorXd& errors) override;
//	void set_triplet_rotations(
//		const Eigen::Ref<const Eigen::Matrix3d>& R12,
//		const Eigen::Ref<const Eigen::Matrix3d>& R13,
//		const Eigen::Ref<const Eigen::Matrix3d>& R23
//	);
//
//private:
//	Eigen::Matrix3d m_R12;
//	Eigen::Matrix3d m_R13;
//	Eigen::Matrix3d m_R23;
//};

class TrilinearKernel final : public RansacKernel<Eigen::Matrix<double, 9, 1>, Eigen::Matrix<double, 9, 1>>
{
public:
	// 通过 RansacKernel 继承
	virtual void run_kernel(
		const std::vector<Eigen::Matrix<double, 9, 1>>& samples, 
		std::vector<Eigen::Matrix<double, 9, 1>>& models
	) override;
	virtual void compute_error(
		const std::vector<Eigen::Matrix<double, 9, 1>>& samples,
		const Eigen::Matrix<double, 9, 1> & model, 
		Eigen::VectorXd & errors
	) override;
	void set_triplet_rotations(
		const Eigen::Ref<const Eigen::Matrix3d>& R12,
		const Eigen::Ref<const Eigen::Matrix3d>& R13,
		const Eigen::Ref<const Eigen::Matrix3d>& R23
	);
	/*void set_triplet_cameras(
		const std::shared_ptr<Camera> camera1,
		const std::shared_ptr<Camera> camera2,
		const std::shared_ptr<Camera> camera3
	);*/

private:
	Eigen::Matrix3d m_R12;
	Eigen::Matrix3d m_R13;
	Eigen::Matrix3d m_R23;

	Eigen::Quaterniond m_q12;
	Eigen::Quaterniond m_q13;
	Eigen::Quaterniond m_q23;

	Eigen::Quaterniond m_q12i;
	Eigen::Quaterniond m_q13i;
	Eigen::Quaterniond m_q23i;

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> m_eig_solver;
	Eigen::Matrix<double, 9, 9> m_mat99;

	/*std::shared_ptr<Camera> m_camera1;
	std::shared_ptr<Camera> m_camera2;
	std::shared_ptr<Camera> m_camera3;*/
};

class TrilinearTranslationsAveraging final : public GlobalTranslationsEstimator
{
public:
	struct Options
	{
		size_t min_num_tracks = 50;				// min tracks per triplet
		size_t min_num_inliers = 30;			// min inliers satisfy the init_max_reproj_error
		double init_max_reproj_error = 5.0;		// max allowed reproject error after triangulate
		double final_max_reproj_error = 2.0;	// max allowed reproject error after bundle adjustment
		bool refine_with_ba = true;
	};

	TrilinearTranslationsAveraging(Options options);

	// 通过 GlobalTranslationsEstimator 继承
	virtual bool estimate(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, ViewCameraBinder & view_camera_binder, EpipolarGraph & epi_graph) override;

private:
	bool solve_linear_system(EpipolarGraph& epi_graph, std::vector<CoinBuild>& lp_builders);
	TripletTranslationsValidator::Options m_validator_options;
	TrilinearTranslationsAveraging::Options m_options;
};
