#pragma once

#include <ceres\ceres.h>
#include <ceres\rotation.h>
#include "Camera.h"

class PinholeDistortK2 final : public Camera
{
public:
	enum {
		OFFSET_FOCAL_X,
		OFFSET_FOCAL_Y,
		OFFSET_PRINCIPAL_X,
		OFFSET_PRINCIPAL_Y,
		OFFSET_RADIAL_K1,
		OFFSET_RADIAL_K2,
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

			T x = pos_proj[0] / pos_proj[2];
			T y = pos_proj[1] / pos_proj[2];
			const T r = x*x + y*y;
			const T r2 = r*r;
			const T r4 = r2*r2;

			const T fx = intrinsic[OFFSET_FOCAL_X];
			const T fy = intrinsic[OFFSET_FOCAL_Y];
			const T cx = intrinsic[OFFSET_PRINCIPAL_X];
			const T cy = intrinsic[OFFSET_PRINCIPAL_Y];
			const T k1 = intrinsic[OFFSET_RADIAL_K1];
			const T k2 = intrinsic[OFFSET_RADIAL_K2];

			const T radial_coeff = T(1.0) + k1 * r2 + k2 * r4;
			x *= radial_coeff;
			y *= radial_coeff;

			// Apply intrinsic
			const T u = fx * x + cx;
			const T v = fy * y + cy;

			residuals[0] = u - T(observation[0]);
			residuals[1] = v - T(observation[1]);

			return true;
		}
	};

	PinholeDistortK2(Eigen::Matrix<double, 6, 1>& intrinsic)
	{
		//Copy intrinsic
		m_intrinsic = intrinsic;
	}

	// Í¨¹ý Camera ¼Ì³Ð
	virtual ceres::CostFunction * create_cost_function(const Eigen::Ref<const Eigen::Vector2d>& observation) override
	{
		return new ceres::AutoDiffCostFunction<PinholeDistortK2::Cost, 2, 6, 6, 3>(
			new PinholeDistortK2::Cost(observation)
			);
	}
	virtual Eigen::Vector3d image_to_world(const Eigen::Ref<const Eigen::Vector2d>& img_point) override
	{

	}
	virtual Eigen::Vector2d world_to_image(const Eigen::Ref<const Eigen::Vector3d>& world_point) override
	{

	}
};



