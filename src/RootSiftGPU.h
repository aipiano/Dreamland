#pragma once

#include "FeatureExtractor.h"
#include <SiftGPU.h>

//------------------
//-- Bibliography --
//------------------
//- [1] "Three things everyone should know to improve object retrieval"
//- Authors: Relja Arandjelovic, Andrew Zisserman
//- Date: 2012.

class RootSiftGPU final : public FeatureExtractor
{
public:
	RootSiftGPU(int first_octave = -1, int num_dog_levels = 3, float dog_thresh = 0.02/3, float edge_thresh = 10.0, int device_id = 0);
	~RootSiftGPU();

	// Í¨¹ý FeatureExtractor ¼Ì³Ð
	virtual bool extract(cv::Mat & image, std::vector<cv::KeyPoint>& keypoints, cv::Mat & descriptors) override;
private:
	bool init_sift_gpu(int first_octave = -1, int num_dog_levels = 3, float dog_thresh = 0.02, float edge_thresh = 10.0, int device_id = 0);
	void free_sift_gpu();

private:
	std::vector<SiftKeypoint> m_sift_keypoints_buf;
	SiftGPU* m_sift = nullptr;
};
