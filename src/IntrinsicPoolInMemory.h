#pragma once

#include <vector>
#include <map>
#include <mutex>
#include "IntrinsicPool.h"

class IntrinsicPoolInMemory final : public IntrinsicPool
{
public:
	IntrinsicPoolInMemory();
	virtual void push(const CameraIntrinsic& intrinsic, int intrinsic_id) override;
	virtual void bind(int intrinsic_id, int view_id) override;
	virtual const CameraIntrinsic& get_intrinsic_by_view(int view_id) override;
	virtual const CameraIntrinsic& get_intrinsic_by_id(int intrinsic_id) override;
	virtual int get_intrinsic_id(int view_id) override;
	virtual size_t size() override;
	virtual void clear() override;

private:
	typedef std::map<int, int> ViewToIntrinsicIDMap;
	typedef std::map<int, CameraIntrinsic> IntrinsicMap;

	std::mutex m_mutex;
	IntrinsicMap m_intrinsics;
	ViewToIntrinsicIDMap m_view_id_to_intrinsic_id;
};