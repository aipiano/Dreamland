#pragma once

#include <ceres\ceres.h>
#include <ceres\rotation.h>
#include <Eigen\Eigen>
#include "Camera.h"

class Pinhole final : public Camera
{
public:
	enum {
		OFFSET_FOCAL_X,
		OFFSET_FOCAL_Y,
		OFFSET_PRINCIPAL_X,
		OFFSET_PRINCIPAL_Y,
		INTRINSIC_LENGTH
	};

	struct Cost
	{
		Eigen::Vector2d observation;

		Cost(const Eigen::Ref<const Eigen::Vector2d>& observation)
			: observation(observation)
		{
		}

		template <typename T>
		bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const
		{
			const T* r = extrinsic;
			const T* t = &extrinsic[3];

			T pos_proj[3];
			ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

			// Apply the camera translation
			pos_proj[0] += t[0];
			pos_proj[1] += t[1];
			pos_proj[2] += t[2];

			const T x = pos_proj[0] / pos_proj[2];
			const T y = pos_proj[1] / pos_proj[2];

			const T fx = intrinsic[OFFSET_FOCAL_X];
			const T fy = intrinsic[OFFSET_FOCAL_Y];
			const T cx = intrinsic[OFFSET_PRINCIPAL_X];
			const T cy = intrinsic[OFFSET_PRINCIPAL_Y];

			// Apply intrinsic
			const T u = fx * x + cx;
			const T v = fy * y + cy;

			residuals[0] = u - T(observation[0]);
			residuals[1] = v - T(observation[1]);

			return true;
		}
	};

	Pinhole(Eigen::Vector4d& intrinsic)
	{
		//Copy intrinsic
		m_intrinsic = intrinsic;
	}

	// Í¨¹ý Camera ¼Ì³Ð
	virtual ceres::CostFunction * create_cost_function(const Eigen::Ref<const Eigen::Vector2d>& observation) override
	{
		return new ceres::AutoDiffCostFunction<Pinhole::Cost, 2, 4, 6, 3>(
			new Pinhole::Cost(observation)
			);
	}
	virtual Eigen::Vector3d image_to_world(const Eigen::Ref<const Eigen::Vector2d>& img_point) override
	{
		const double fx = m_intrinsic[OFFSET_FOCAL_X];
		const double fy = m_intrinsic[OFFSET_FOCAL_Y];
		const double cx = m_intrinsic[OFFSET_PRINCIPAL_X];
		const double cy = m_intrinsic[OFFSET_PRINCIPAL_Y];

		return Eigen::Vector3d((img_point[0] - cx) / fx, (img_point[1] - cy) / fy, 1.0);
	}
	virtual Eigen::Vector2d world_to_image(const Eigen::Ref<const Eigen::Vector3d>& world_point) override
	{
		const double fx = m_intrinsic[OFFSET_FOCAL_X];
		const double fy = m_intrinsic[OFFSET_FOCAL_Y];
		const double cx = m_intrinsic[OFFSET_PRINCIPAL_X];
		const double cy = m_intrinsic[OFFSET_PRINCIPAL_Y];

		Eigen::Vector3d w = world_point / world_point[2];
		return Eigen::Vector2d(fx * w[0] + cx, fy * w[1] + cy);
	}
};
