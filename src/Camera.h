#pragma once

#include <ceres\ceres.h>
#include <Eigen\Eigen>
#include "CameraIntrinsic.h"

class Camera
{
public:
	virtual ceres::CostFunction* create_cost_function(const Eigen::Ref<const Eigen::Vector2d>& observation) = 0;
	virtual CameraIntrinsic& get_intrinsic() { return m_intrinsic; }
	virtual Eigen::Vector3d image_to_world(const Eigen::Ref<const Eigen::Vector2d>& img_point) = 0;
	virtual Eigen::Vector2d world_to_image(const Eigen::Ref<const Eigen::Vector3d>& world_point) = 0;

protected:
	CameraIntrinsic m_intrinsic;
};
