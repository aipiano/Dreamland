#pragma once

#include <Eigen\Eigen>
#include "CameraExtrinsic.h"

struct GlobalTransform
{
	// 从世界坐标系到该相机坐标系的变换
	CameraExtrinsic rt;

	bool is_inlier = false;
	int view_id = -1;
	//size_t inner_idx = 0;
};
