#pragma once

#include <Eigen\Eigen>
#include "CameraExtrinsic.h"

struct GlobalTransform
{
	// ����������ϵ�����������ϵ�ı任
	CameraExtrinsic rt;

	bool is_inlier = false;
	int view_id = -1;
	//size_t inner_idx = 0;
};
