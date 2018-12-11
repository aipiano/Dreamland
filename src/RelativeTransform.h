#pragma once

#include <Eigen\Eigen>
#include "CameraExtrinsic.h"

struct RelativeTransform
{
	CameraExtrinsic rt;

	//指明变换的方向
	int src_id = -1;
	int dst_id = -1;

	bool is_inlier = false;
	double weight = 0.0;
	//size_t inner_idx = 0;
};
