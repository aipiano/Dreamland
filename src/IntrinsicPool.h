#pragma once

#include <vector>
#include "CameraIntrinsic.h"

class IntrinsicPool
{
public:
	virtual void push(const CameraIntrinsic& intrinsic, int intrinsic_id) = 0;
	virtual void bind(int intrinsic_id, int view_id) = 0;
	virtual const CameraIntrinsic& get_intrinsic_by_view(int view_id) = 0;
	virtual const CameraIntrinsic& get_intrinsic_by_id(int intrinsic_id) = 0;
	virtual int get_intrinsic_id(int view_id) = 0;
	virtual size_t size() = 0;
	virtual void clear() = 0;
};