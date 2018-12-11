#include "IntrinsicPoolInMemory.h"

IntrinsicPoolInMemory::IntrinsicPoolInMemory()
{
}

void IntrinsicPoolInMemory::push(const CameraIntrinsic & intrinsic, int intrinsic_id)
{
	m_mutex.lock();
	m_intrinsics[intrinsic_id] = intrinsic;
	m_mutex.unlock();
}

void IntrinsicPoolInMemory::bind(int intrinsic_id, int view_id)
{
	m_mutex.lock();
	m_view_id_to_intrinsic_id[view_id] = intrinsic_id;
	m_mutex.unlock();
}

const CameraIntrinsic & IntrinsicPoolInMemory::get_intrinsic_by_view(int view_id)
{
	return m_intrinsics[m_view_id_to_intrinsic_id[view_id]];
}

const CameraIntrinsic & IntrinsicPoolInMemory::get_intrinsic_by_id(int intrinsic_id)
{
	return m_intrinsics[intrinsic_id];
}

int IntrinsicPoolInMemory::get_intrinsic_id(int view_id)
{
	return m_view_id_to_intrinsic_id[view_id];
}

size_t IntrinsicPoolInMemory::size()
{
	return m_intrinsics.size();
}

void IntrinsicPoolInMemory::clear()
{
	m_mutex.lock();
	m_intrinsics.clear();
	m_view_id_to_intrinsic_id.clear();
	m_mutex.unlock();
}

