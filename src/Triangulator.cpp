#include "Triangulator.h"
#include "Utils.h"
#include "Ransac.hpp"

using namespace std;
using namespace Eigen;

size_t Triangulator::two_views_l2(Scene & scene, bool retriangulate, double max_reproj_error, double min_baseline_angle_in_degree)
{
	return size_t();
}

size_t Triangulator::multi_views_dlt(Scene& scene, bool retriangulate, double max_reproj_error, double min_baseline_angle_in_degree, int min_inlier_nodes_per_track)
{
	Matrix<double, 3, 4> P;
	Vector3d x;
	Vector4d X;
	auto& views = scene.views;
	SelfAdjointEigenSolver<Matrix4d> eig_solver;
	size_t num_inlier_tracks = 0;

	// DLT方法进行三角化
	// TODO: 多线程
	for (auto& track : scene.tracks)
	{
		if (retriangulate)
		{
			if (track.is_inlier)
			{
				// if track is an inlier, we just check it again. do not triangulate it.
				/*if (check_reproject_errors(scene, track, max_reproj_error, min_inlier_nodes_per_track) &&
					check_baseline_angles(scene, track, min_baseline_angle_in_degree))
				{
					++num_inlier_tracks;
				}
				else
				{
					track.is_inlier = false;
				}*/
				++num_inlier_tracks;
				continue;
			}
			// if track is an outlier, we do triangulation.
		}
		/*else
		{
			if (!track.is_inlier) continue;
		}*/

		// if retriangulate == false, do triangulation despite of the track is an inlier or an outlier.

		// triangulate track
		auto& nodes = track.nodes;
		Matrix4d AtA = Matrix4d::Zero();
		Matrix<double, 3, 4> xP;
		for (size_t n = 0; n < nodes.size(); ++n)
		{
			auto& view = views[scene.view_idx_by_view_id[nodes[n].view_id]];
			auto& camera = scene.view_camera_binder.get_camera(view.view_id);

			angle_axis_to_rotation_matrix(view.rt.topRows(3), P.block(0, 0, 3, 3));
			P.rightCols(1) = view.rt.bottomRows(3);
			x = camera->image_to_world(nodes[n].observation);
			xP.noalias() = cross_mat(x) * P;
			AtA += xP.transpose() * xP;
		}
		eig_solver.compute(AtA);
		// eigen values are stored in ascending order. so the first eigen vector is correspond to the smallest eigen value.
		X = eig_solver.eigenvectors().col(0);

		if (abs(X[3]) < DBL_EPSILON)
		{
			track.is_inlier = false;
			continue;
		}
		track.point3d = (X / X[3]).topRows(3);

		// check the result
		if (check_reproject_errors(scene, track, max_reproj_error, min_inlier_nodes_per_track) &&
			check_baseline_angles(scene, track, min_baseline_angle_in_degree))
		{
			track.is_inlier = true;
			++num_inlier_tracks;
		}
		else
		{
			track.is_inlier = false;
		}
	}

	return num_inlier_tracks;
}

size_t Triangulator::multi_views_robust(Scene & scene, bool retriangulate, double max_reproj_error, double min_baseline_angle_in_degree, int min_inlier_nodes_per_track)
{
	std::shared_ptr<RobustTriangulationKernel> kernel = make_shared<RobustTriangulationKernel>();
	Ransac<NodeCameraPair, Eigen::Vector4d> ransac(kernel, 2, max_reproj_error, 0.999, 16);

	auto& views = scene.views;
	vector<NodeCameraPair> samples;
	Vector4d X;
	vector<unsigned char> mask;
	size_t num_inlier_tracks = 0;
	for (auto& track : scene.tracks)
	{
		if (retriangulate)
		{
			if (track.is_inlier)
			{
				// if track is an inlier, we just check it again. do not triangulate it.
				/*if (check_reproject_errors(scene, track, max_reproj_error, min_inlier_nodes_per_track) &&
					check_baseline_angles(scene, track, min_baseline_angle_in_degree))
				{
					++num_inlier_tracks;
				}
				else
				{
					track.is_inlier = false;
				}*/
				++num_inlier_tracks;
				continue;
			}
			// if track is an outlier, we do triangulation.
		}

		auto& nodes = track.nodes;
		size_t nodes_size = nodes.size();
		samples.resize(nodes_size);
		for (size_t i = 0; i < nodes_size; ++i)
		{
			auto& view = views[scene.view_idx_by_view_id[nodes[i].view_id]];
			auto& camera = scene.view_camera_binder.get_camera(view.view_id);

			samples[i].first.topRows(2) = nodes[i].observation;
			samples[i].first.bottomRows(6) = view.rt;
			samples[i].second = camera;
		}
		if (!ransac.run(samples, X, mask))
		{
			track.is_inlier = false;
			continue;
		}

		int num_inlier_nodes = 0;
		for (size_t i = 0; i < nodes_size; ++i)
		{
			nodes[i].is_inlier = mask[i];
			num_inlier_nodes += mask[i];
		}
		if (num_inlier_nodes < min_inlier_nodes_per_track || 
			double(num_inlier_nodes) / nodes_size < 0.6)
		{
			track.is_inlier = false;
			continue;
		}

		// reestimate point3d by all inliers.
		if (num_inlier_nodes > 2)
		{
			vector<NodeCameraPair> inliers;
			vector<Vector4d> models;
			inliers.reserve(num_inlier_nodes);
			for (size_t i = 0; i < nodes_size; ++i)
			{
				if (!nodes[i].is_inlier)
					continue;
				inliers.emplace_back(samples[i]);
			}
			kernel->run_kernel(inliers, models);
			X = models[0];
		}
		track.point3d = (X / X[3]).topRows(3);

		// no need to check reproject errors, we did that in ransac procedure.
		if (check_baseline_angles(scene, track, min_baseline_angle_in_degree))
		{
			track.is_inlier = true;
			++num_inlier_tracks;
		}
		else
		{
			track.is_inlier = false;
		}
	}

	return num_inlier_tracks;
}

bool Triangulator::check_reproject_errors(Scene & scene, Track & track, double max_reproj_error, int min_inlier_nodes_per_track)
{
	// compute reprojected error
	double reproj_error = 0;
	int num_inlier_nodes = 0;
	auto& nodes = track.nodes;
	auto& views = scene.views;
	Matrix3d R;
	Vector3d X;
	size_t node_size = nodes.size();

	// calculate reproject errors
	for (size_t i = 0; i < node_size; ++i)
	{
		auto& view_i = views[scene.view_idx_by_view_id[nodes[i].view_id]];
		auto& camera = scene.view_camera_binder.get_camera(view_i.view_id);

		angle_axis_to_rotation_matrix(view_i.rt.topRows(3), R);
		X = R * track.point3d + view_i.rt.bottomRows(3);
		if (X[2] <= 0)	// the point must in front of the cameras which could see it.
		{
			return false;
		}
		reproj_error = (camera->world_to_image(X) - nodes[i].observation).norm();
		if (reproj_error > max_reproj_error)
		{
			nodes[i].is_inlier = false;
		}
		else
		{
			nodes[i].is_inlier = true;
			++num_inlier_nodes;
		}
	}
	double inlier_ratio = double(num_inlier_nodes) / node_size;
	if (num_inlier_nodes < min_inlier_nodes_per_track || 
		inlier_ratio < 0.6)
	{
		return false;
	}
	return true;
}

bool Triangulator::check_baseline_angles(Scene & scene, Track & track, double min_baseline_angle_in_degree)
{
	double max_baseline_angle = 0;
	auto& nodes = track.nodes;
	auto& views = scene.views;
	size_t node_size = nodes.size();

	// calculate baseline angle
	for (size_t i = 0; i < node_size; ++i)
	{
		if (!nodes[i].is_inlier) continue;

		auto& view_i = views[scene.view_idx_by_view_id[nodes[i].view_id]];
		Vector3d ray_i = track.point3d - view_i.rt.bottomRows(3);

		for (size_t j = i + 1; j < node_size; ++j)
		{
			if (!nodes[j].is_inlier) continue;

			auto& view_j = views[scene.view_idx_by_view_id[nodes[j].view_id]];
			Vector3d ray_j = track.point3d - view_j.rt.bottomRows(3);
			double angle = rad2deg(angle_between(ray_i, ray_j));
			max_baseline_angle = max(angle, max_baseline_angle);
		}
	}
	if (max_baseline_angle < min_baseline_angle_in_degree)
	{
		return false;
	}
	return true;
}



//////////////////////////////////////////////////////////////////////
//					RobustTriangulationKernel
//////////////////////////////////////////////////////////////////////

void RobustTriangulationKernel::run_kernel(
	const std::vector<NodeCameraPair>& samples,
	std::vector<Eigen::Vector4d> & models
)
{
	static Matrix4d AtA;
	static Matrix<double, 3, 4> xP;
	static Matrix<double, 3, 4> P;
	static Vector3d x;
	static SelfAdjointEigenSolver<Matrix4d> eig_solver;

	models.clear();
	if (samples.size() == 0) return;

	// samples stored observed points, extrinsics and camera.
	size_t num_views = samples.size();
	AtA.setZero();
	for (size_t i = 0; i < num_views; ++i)
	{
		auto& sample = samples[i].first;
		auto& camera = *samples[i].second;

		x = camera.image_to_world(sample.topRows(2));
		angle_axis_to_rotation_matrix(sample.middleRows(2, 3), P.block(0, 0, 3, 3));
		P.rightCols(1) = sample.bottomRows(3);
		xP.noalias() = cross_mat(x) * P;
		AtA += xP.transpose() * xP;
	}

	eig_solver.compute(AtA);
	models.resize(1);
	// eigen values are stored in ascending order. so the first eigen vector is correspond to the smallest eigen value.
	models[0] = eig_solver.eigenvectors().col(0);
}

void RobustTriangulationKernel::compute_error(
	const std::vector<NodeCameraPair>& samples,
	const Eigen::Vector4d& model,
	Eigen::VectorXd & errors
)
{
	size_t num_views = samples.size();
	errors.resize(num_views);

	if (abs(model[3]) < DBL_EPSILON)
	{
		errors.fill(DBL_MAX);
		return;
	}

	Matrix3d R;
	Vector3d X;
	Vector3d point3d = (model / model[3]).topRows(3);
	for (size_t i = 0; i < num_views; ++i)
	{
		auto& sample = samples[i].first;
		auto& camera = *samples[i].second;

		angle_axis_to_rotation_matrix(sample.middleRows(2, 3), R);
		X = R * point3d + sample.bottomRows(3);
		if (X[2] <= 0)	// the point must in front of the cameras which could see it.
		{
			errors.fill(DBL_MAX);
			return;
		}

		errors[i] = (camera.world_to_image(X) - sample.topRows(2)).norm();
	}
}
