#include "FeaturePoolInMemory.h"

using namespace std;
using namespace cv;

void FeaturePoolInMemory::push(const KeyPoints& keypoints, const cv::Mat & descriptors, int view_id)
{
	KeyPointsPtr kps = make_shared<KeyPoints>(keypoints);
	DescriptorsPtr desc = make_shared<Mat>(descriptors);

	m_mutex.lock();
	m_keypoints_map[view_id] = kps;
	m_descriptors_map[view_id] = desc;
	m_mutex.unlock();
}

void FeaturePoolInMemory::push(KeyPoints&& keypoints, cv::Mat && descriptors, int view_id)
{
	KeyPointsPtr kps = make_shared<KeyPoints>(std::move(keypoints));
	DescriptorsPtr desc = make_shared<Mat>(std::move(descriptors));

	m_mutex.lock();
	m_keypoints_map[view_id] = kps;
	m_descriptors_map[view_id] = desc;
	m_mutex.unlock();
}

const FeaturePoolInMemory::KeyPointsPtr FeaturePoolInMemory::get_keypoints(int view_id)
{
	auto find_result = m_keypoints_map.find(view_id);
	if (find_result == m_keypoints_map.end())
	{
		return nullptr;
	}
	return find_result->second;
}

const FeaturePoolInMemory::DescriptorsPtr FeaturePoolInMemory::get_descriptors(int view_id)
{
	auto find_result = m_descriptors_map.find(view_id);
	if (find_result == m_descriptors_map.end())
	{
		return nullptr;
	}
	return find_result->second;
}

size_t FeaturePoolInMemory::size()
{
	return m_keypoints_map.size();
}

void FeaturePoolInMemory::clear()
{
	m_mutex.lock();
	m_keypoints_map.clear();
	m_descriptors_map.clear();
	m_mutex.unlock();
}
