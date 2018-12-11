#pragma once

#include "Scene.h"
#include "RansacKernel.hpp"
#include <vector>
#include <map>
#include <Eigen/Eigen>

// if retriangulate == false, it will triangulate all tracks. otherwise, it will triangulate outlier tracks only
class Triangulator
{
public:
	size_t two_views_l2(Scene& scene, bool retriangulate = false, double max_reproj_error = 5., double min_baseline_angle_in_degree = 3.);
	size_t multi_views_dlt(Scene& scene, bool retriangulate = false, double max_reproj_error = 5., double min_baseline_angle_in_degree = 3., int min_inlier_nodes_per_track = 2);
	size_t multi_views_robust(Scene& scene, bool retriangulate = false, double max_reproj_error = 5., double min_baseline_angle_in_degree = 3., int min_inlier_nodes_per_track = 2);

private:
	bool check_reproject_errors(Scene& scene, Track& track, double max_reproj_error, int min_inlier_nodes_per_track);
	bool check_baseline_angles(Scene& scene, Track& track, double min_baseline_angle_in_degree);
};

typedef std::pair<Eigen::Matrix<double, 8, 1>, std::shared_ptr<Camera>> NodeCameraPair;

class RobustTriangulationKernel final : public RansacKernel<NodeCameraPair, Eigen::Vector4d>
{
public:
	// Í¨¹ý RansacKernel ¼Ì³Ð
	virtual void run_kernel(
		const std::vector<NodeCameraPair>& samples,
		std::vector<Eigen::Vector4d> & models
	) override;
	virtual void compute_error(
		const std::vector<NodeCameraPair>& samples,
		const Eigen::Vector4d& model, 
		Eigen::VectorXd & errors
	) override;
};
