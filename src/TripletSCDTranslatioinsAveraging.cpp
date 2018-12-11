#include "TripletSCDTranslatioinsAveraging.h"
#include "TracksBuilder.h"
#include "TripletsBuilder.h"
#include "Utils.h"
#include <ClpSimplex.hpp>
#include <CoinBuild.hpp>
#include <CoinPackedVector.hpp>
#include <Eigen/Eigen>
#include <omp.h>
#include <ceres\rotation.h>

using namespace std;
using namespace boost;
using namespace Eigen;

struct ChordalDistanceV2V	// View-to-View
{
	double weight = 1.0;
	Vector3d world_direction;	// normalized direction in the world coordinate which targeting view2 from view1.
	Vector3d view1_rotation;
	Vector3d view2_rotation;

	ChordalDistanceV2V(
		const Eigen::Ref<const Vector3d>& view1_rotation, 
		const Eigen::Ref<const Vector3d>& view2_rotation, 
		const Eigen::Ref<const Vector3d>& world_direction,
		double weight = 1.0
	)
		: view1_rotation(view1_rotation), view2_rotation(view2_rotation), world_direction(world_direction), weight(weight)
	{
	}

	template <typename T>
	bool operator()(const T* const view1_trans, const T* const view2_trans, T* residuals) const
	{
		T view1_rot[3];
		T view2_rot[3];

		// 将view_trans变换到世界坐标系，首先取相机旋转变换的逆变换
		view1_rot[0] = T(-view1_rotation[0]);
		view1_rot[1] = T(-view1_rotation[1]);
		view1_rot[2] = T(-view1_rotation[2]);

		view2_rot[0] = T(-view2_rotation[0]);
		view2_rot[1] = T(-view2_rotation[1]);
		view2_rot[2] = T(-view2_rotation[2]);

		T view1_pos[3];
		T view2_pos[3];
		ceres::AngleAxisRotatePoint(view1_rot, view1_trans, view1_pos);
		ceres::AngleAxisRotatePoint(view2_rot, view2_trans, view2_pos);

		T dir[3];
		// rotated_trans乘以-1后才是view的空间坐标
		dir[0] = -view2_pos[0] + view1_pos[0];
		dir[1] = -view2_pos[1] + view1_pos[1];
		dir[2] = -view2_pos[2] + view1_pos[2];

		// normalize
		T length = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
		length = ceres::sqrt(length) + T(DBL_EPSILON);
		dir[0] /= length;
		dir[1] /= length;
		dir[2] /= length;

		// calculate residuals
		residuals[0] = dir[0] - T(world_direction[0]);
		residuals[1] = dir[1] - T(world_direction[1]);
		residuals[2] = dir[2] - T(world_direction[2]);

		// apply weight
		residuals[0] *= T(weight);
		residuals[1] *= T(weight);
		residuals[2] *= T(weight);

		return true;
	}
};

struct ChordalDistanceV2P	// View-to-Point
{
	double weight = 1.0;
	Vector3d world_direction;	// normalized direction in the world coordinate which targeting point from view.
	Vector3d view_rotation;

	ChordalDistanceV2P(
		const Eigen::Ref<const Vector3d>& view_rotation, 
		const Eigen::Ref<const Vector3d>& world_direction,
		double weight = 1.0)
		: view_rotation(view_rotation), world_direction(world_direction), weight(weight)
	{}

	template <typename T>
	bool operator()(const T* const view_trans, const T* const point_pos, T* residuals) const
	{
		T view_rot[3];
		// 将view_trans变换到世界坐标系，首先取相机旋转变换的逆变换
		view_rot[0] = T(-view_rotation[0]);
		view_rot[1] = T(-view_rotation[1]);
		view_rot[2] = T(-view_rotation[2]);

		T view_pos[3];
		ceres::AngleAxisRotatePoint(view_rot, view_trans, view_pos);

		T dir[3];
		// dir[0] = point_pos[0] - (-view_pos[0]);
		dir[0] = point_pos[0] + view_pos[0];
		dir[1] = point_pos[1] + view_pos[1];
		dir[2] = point_pos[2] + view_pos[2];

		// normalize
		T length = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
		length = ceres::sqrt(length) + T(DBL_EPSILON);
		dir[0] /= length;
		dir[1] /= length;
		dir[2] /= length;

		// calculate residuals
		residuals[0] = dir[0] - T(world_direction[0]);
		residuals[1] = dir[1] - T(world_direction[1]);
		residuals[2] = dir[2] - T(world_direction[2]);

		// apply weight
		residuals[0] *= T(weight);
		residuals[1] *= T(weight);
		residuals[2] *= T(weight);

		return true;
	}
};


bool TripletSCDTranslationsAveraging::estimate(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, ViewCameraBinder & view_camera_binder, EpipolarGraph & epi_graph)
{
	TripletsBuilder triplet_builder;
	Triplets triplets;
	triplet_builder.build(epi_graph);
	triplet_builder.swap(triplets);

	size_t num_vertex = num_vertices(epi_graph);
	size_t cols = 3 * num_vertex + triplets.size() + 1;
	vector<int> cam_triple_count(num_vertex, 0);

	ceres::Solver::Options ceres_options;
	set_ceres_options(ceres_options);

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
	for (int triplet_idx = 0; triplet_idx < triplets_count; ++triplet_idx)
	{
		auto& t = triplets[triplet_idx];
#ifdef _DEBUG
		int thread_number = 0;
#else
		int thread_number = omp_get_thread_num();
#endif
		CoinBuild& lp_builder = lp_builders[thread_number];

		Scene tiny_scene;
		build_scene_from_triplet(t, feature_pool, match_pool, view_camera_binder, epi_graph, tiny_scene);
		if (tiny_scene.tracks.size() < 6)
			continue;

		init_points3d_randomly(tiny_scene.tracks);

		double mse = solve_scene(tiny_scene, ceres_options);
		if (mse > 0.1) continue;


	}

	return true;
}

void TripletSCDTranslationsAveraging::build_scene_from_triplet(
	Triplet & triplet,
	std::shared_ptr<FeaturePool> feature_pool,
	std::shared_ptr<MatchPool> match_pool,
	ViewCameraBinder & view_camera_binder,
	EpipolarGraph & epi_graph,
	Scene & scene
)
{
	// must have idx_i < idx_j < idx_k;
	int idx_i = triplet.e_ij.m_source;
	int idx_j = triplet.e_ij.m_target;
	int idx_k = triplet.e_ik.m_target;

	auto view_id1 = epi_graph[idx_i].view_id;
	auto view_id2 = epi_graph[idx_j].view_id;
	auto view_id3 = epi_graph[idx_k].view_id;

	auto camera1 = view_camera_binder.get_camera(view_id1);
	auto camera2 = view_camera_binder.get_camera(view_id2);
	auto camera3 = view_camera_binder.get_camera(view_id3);

	GlobalTransform P1;
	P1.is_inlier = true;
	P1.view_id = view_id1;
	P1.rt = CameraExtrinsic::Zero();

	auto& transform_ij = epi_graph[triplet.e_ij];
	GlobalTransform P2;
	P2.is_inlier = true;
	P2.view_id = view_id2;
	// e_ij的两个端点对应view_id1和view_id2，不过变换方向可能从view_id1到view_id2，也可能相反，所以要判断并统一为view_id1到view_id2
	P2.rt = (transform_ij.src_id == view_id1 ? transform_ij.rt : reversed_transform(transform_ij.rt));

	auto& transform_ik = epi_graph[triplet.e_ik];
	GlobalTransform P3;
	P3.is_inlier = true;
	P3.view_id = view_id3;
	// 同上
	P3.rt = (transform_ik.src_id == view_id1 ? transform_ik.rt : reversed_transform(transform_ik.rt));

	scene.views.clear();
	scene.views.push_back(P1);	// 0
	scene.views.push_back(P2);	// 1
	scene.views.push_back(P3);	// 2

	scene.view_idx_by_view_id.clear();
	scene.view_idx_by_view_id[view_id1] = 0;	// P1
	scene.view_idx_by_view_id[view_id2] = 1;	// P2
	scene.view_idx_by_view_id[view_id3] = 2;	// P3

	scene.view_camera_binder.clear();
	scene.view_camera_binder.bind(view_id1, camera1);
	scene.view_camera_binder.bind(view_id2, camera2);
	scene.view_camera_binder.bind(view_id3, camera3);

	PairwiseMatches triple_pairs;
	triple_pairs.reserve(2);
	triple_pairs.push_back(PairwiseMatch(view_id1, view_id2));
	triple_pairs.push_back(PairwiseMatch(view_id1, view_id3));

	TracksBuilder tracks_builder;
	tracks_builder.build(feature_pool, match_pool, triple_pairs);
	tracks_builder.filter(3);

	scene.tracks.clear();
	tracks_builder.swap(scene.tracks);
}

void TripletSCDTranslationsAveraging::set_ceres_options(ceres::Solver::Options & options)
{
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.preconditioner_type = ceres::JACOBI;
	options.minimizer_progress_to_stdout = false;
	options.logging_type = ceres::SILENT;
	options.num_threads = 1;
	options.num_linear_solver_threads = 1;
	options.function_tolerance = 1e-6;
	options.gradient_tolerance = 1e-10;
	options.parameter_tolerance = 1e-8;
	options.max_num_iterations = 500;
	options.max_solver_time_in_seconds = 30;
	options.use_inner_iterations = true;
}

double TripletSCDTranslationsAveraging::solve_scene(Scene & scene, ceres::Solver::Options & options)
{
	ceres::Problem problem;

	// initialize problem
	//add views
	for (auto& view : scene.views)
	{
		problem.AddParameterBlock(view.rt.bottomRows(3).data(), 3);
	}
	// Fix the first extrinsic
	problem.SetParameterBlockConstant(scene.views[0].rt.bottomRows(3).data());

	// add view-to-point residuals
	ceres::LossFunction* loss_function = new ceres::HuberLoss(std::sqrt(0.1));
	auto& view_idx_by_view_id = scene.view_idx_by_view_id;
	double weight_v2p = 3 / (scene.tracks.size() * 3);
	//int num_tracks = 0;
	for (auto& track : scene.tracks)
	{
		if (!track.is_inlier)
			continue;

		//if (num_tracks >= 10) break;
		//num_tracks++;

		for (auto& node : track.nodes)
		{
			if (!node.is_inlier)
				continue;

			GlobalTransform& view = scene.views[view_idx_by_view_id[node.view_id]];
			auto camera = scene.view_camera_binder.get_camera(node.view_id);

			Matrix3d R;
			angle_axis_to_rotation_matrix(view.rt.topRows(3), R);
			Vector3d direction = R.transpose() * camera->image_to_world(node.observation);
			direction.normalize();

			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ChordalDistanceV2P, 3, 3, 3>(
				new ChordalDistanceV2P(view.rt.topRows(3), direction, weight_v2p)
				);

			if (cost_function == nullptr) continue;
			problem.AddResidualBlock(
				cost_function,
				loss_function,
				view.rt.bottomRows(3).data(),	// View Translation
				track.point3d.data()			// Point in 3D space
			);
		}
	}

	// add view-to-view residuals
	Matrix3d R;
	angle_axis_to_rotation_matrix(scene.views[1].rt.topRows(3), R);
	Vector3d direction_01 = -R.transpose() * scene.views[1].rt.bottomRows(3);
	direction_01.normalize();
	ceres::CostFunction* cost_function_01 = new ceres::AutoDiffCostFunction<ChordalDistanceV2V, 3, 3, 3>(
		new ChordalDistanceV2V(scene.views[0].rt.topRows(3), scene.views[1].rt.topRows(3), direction_01)
		);
	problem.AddResidualBlock(
		cost_function_01,
		loss_function,
		scene.views[0].rt.bottomRows(3).data(),
		scene.views[1].rt.bottomRows(3).data()
	);

	angle_axis_to_rotation_matrix(scene.views[2].rt.topRows(3), R);
	Vector3d direction_02 = -R.transpose() * scene.views[2].rt.bottomRows(3);
	direction_02.normalize();
	ceres::CostFunction* cost_function_02 = new ceres::AutoDiffCostFunction<ChordalDistanceV2V, 3, 3, 3>(
		new ChordalDistanceV2V(scene.views[0].rt.topRows(3), scene.views[2].rt.topRows(3), direction_02)
		);
	problem.AddResidualBlock(
		cost_function_02,
		loss_function,
		scene.views[0].rt.bottomRows(3).data(),
		scene.views[2].rt.bottomRows(3).data()
	);

	Vector3d direction_12 = direction_02 - direction_01;
	direction_12.normalize();
	ceres::CostFunction* cost_function_12 = new ceres::AutoDiffCostFunction<ChordalDistanceV2V, 3, 3, 3>(
		new ChordalDistanceV2V(scene.views[1].rt.topRows(3), scene.views[2].rt.topRows(3), direction_12)
		);
	problem.AddResidualBlock(
		cost_function_12,
		loss_function,
		scene.views[1].rt.bottomRows(3).data(),
		scene.views[2].rt.bottomRows(3).data()
	);

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	if (!summary.IsSolutionUsable())
		return numeric_limits<double>::max();

	double final_rmse = std::sqrt(summary.final_cost / summary.num_residuals);
	return final_rmse;
}

void TripletSCDTranslationsAveraging::init_points3d_randomly(Tracks & tracks)
{
	for (auto& track : tracks)
	{
		track.point3d.setRandom();
	}
}
