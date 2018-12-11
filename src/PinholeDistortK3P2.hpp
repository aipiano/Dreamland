#pragma once

#include <ceres\ceres.h>
#include <ceres\rotation.h>
#include "Camera.h"

class PinholeDistortK3P2 final : public Camera
{
public:
	enum {
		OFFSET_FOCAL_X,
		OFFSET_FOCAL_Y,
		OFFSET_PRINCIPAL_X,
		OFFSET_PRINCIPAL_Y,
		OFFSET_RADIAL_K1,
		OFFSET_RADIAL_K2,
		OFFSET_TANGENT_P1,
		OFFSET_TANGENT_P2,
		OFFSET_RADIAL_K3,
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
			const T* vec_r = extrinsic;
			const T* t = &extrinsic[3];

			T pos_proj[3];
			ceres::AngleAxisRotatePoint(vec_r, pos3d, pos_proj);

			// Apply the camera translation
			pos_proj[0] += t[0];
			pos_proj[1] += t[1];
			pos_proj[2] += t[2];

			const T ox = pos_proj[0] / pos_proj[2];
			const T oy = pos_proj[1] / pos_proj[2];
			const T r = ox*ox + oy*oy;
			const T r2 = r*r;
			const T r4 = r2*r2;
			const T r6 = r2*r4;

			const T fx = intrinsic[OFFSET_FOCAL_X];
			const T fy = intrinsic[OFFSET_FOCAL_Y];
			const T cx = intrinsic[OFFSET_PRINCIPAL_X];
			const T cy = intrinsic[OFFSET_PRINCIPAL_Y];
			const T k1 = intrinsic[OFFSET_RADIAL_K1];
			const T k2 = intrinsic[OFFSET_RADIAL_K2];
			const T p1 = intrinsic[OFFSET_TANGENT_P1];
			const T p2 = intrinsic[OFFSET_TANGENT_P2];
			const T k3 = intrinsic[OFFSET_RADIAL_K3];

			const T radial_coeff = T(1.0) + k1 * r2 + k2 * r4 + k3 * r6;
			const T x = ox*radial_coeff + T(2.0)*p1*ox*oy + p2*(r2 + T(2.0)*ox*ox);
			const T y = oy*radial_coeff + p1*(r2 + T(2.0)*oy*oy) + T(2.0)*p2*ox*oy;

			// Apply intrinsic
			const T u = fx * x + cx;
			const T v = fy * y + cy;

			residuals[0] = u - T(observation[0]);
			residuals[1] = v - T(observation[1]);

			return true;
		}
	};

	PinholeDistortK3P2(Eigen::Matrix<double, 9, 1>& intrinsic)
	{
		//Copy intrinsic
		m_intrinsic = intrinsic;
	}

	// Í¨¹ý Camera ¼Ì³Ð
	virtual ceres::CostFunction * create_cost_function(const Eigen::Ref<const Eigen::Vector2d>& observation) override
	{
		return new ceres::AutoDiffCostFunction<PinholeDistortK3P2::Cost, 2, 9, 6, 3>(
			new PinholeDistortK3P2::Cost(observation)
			);
	}
	virtual Eigen::Vector3d image_to_world(const Eigen::Ref<const Eigen::Vector2d>& img_point) override
	{

	}
	virtual Eigen::Vector2d world_to_image(const Eigen::Ref<const Eigen::Vector3d>& world_point) override
	{

	}
};


