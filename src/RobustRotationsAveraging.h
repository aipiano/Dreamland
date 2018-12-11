#pragma once

#include "GlobalRotationsEstimator.h"
#include "Utils.h"
#include <Eigen\Eigen>
#include <Eigen\Sparse>

/*
Based on: Efficient and Robust Large-Scale Rotation Averaging.
*/
class RobustRotationsAveraging final : public GlobalRotationsEstimator
{
public:
	struct Options
	{
		double sigma = deg2rad(5);			// in degree
		double max_difference_degree = 5.0;	// max difference angle in degree between global roations and relative rotations
	};

	RobustRotationsAveraging(Options options);

	// Í¨¹ý GlobalRotationsEstimator ¼Ì³Ð
	virtual bool estimate(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, ViewCameraBinder & view_camera_binder, EpipolarGraph & epi_graph) override;

private:
	Options m_options;

private:
	void initialize_system(
		const EpipolarGraph& epi_graph, 
		std::vector<Eigen::Matrix3d>& R, 
		std::vector<Eigen::Matrix3d>& Rij, 
		Eigen::SparseMatrix<double, Eigen::RowMajor>& A
	);

	void calc_rotation_errors(
		const EpipolarGraph& epi_graph, 
		const std::vector<Eigen::Matrix3d>& R,
		const std::vector<Eigen::Matrix3d>& Rij,
		Eigen::VectorXd& delta_r_ij
	);
	void update_global_rotations(
		const Eigen::VectorXd& delta_r,
		std::vector<Eigen::Matrix3d>& R
	);

	bool l1_averaging(
		const EpipolarGraph& epi_graph,
		const std::vector<Eigen::Matrix3d>& Rij,
		const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
		std::vector<Eigen::Matrix3d>& R
	);
	bool solve_l1(
		const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
		const Eigen::VectorXd& delta_r_ij,
		Eigen::VectorXd& delta_r
	);

	bool irls_averaging(
		const EpipolarGraph& epi_graph,
		const std::vector<Eigen::Matrix3d>& Rij,
		const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
		std::vector<Eigen::Matrix3d>& R
	);
	inline void build_weights(const Eigen::VectorXd& residuals, Eigen::VectorXd& weights);

	void update_graph(EpipolarGraph& epi_graph, const std::vector<Eigen::Matrix3d>& R);
	void filter_outlier_edges(EpipolarGraph& epi_graph, const std::vector<Eigen::Matrix3d>& Rij);
};
