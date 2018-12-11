#pragma once

#include "FeatureMatcher.h"
#include "FeatureMatcher.h"
#include "FeaturePool.h"
#include <memory>

class FlannMatcher : public FeatureMatcher
{
public:
	FlannMatcher(std::shared_ptr<FeaturePool> features, float distinct_ratio = 0.6f);

	// Í¨¹ý FeatureMatcher ¼Ì³Ð
	virtual void train(int train_id) override;
	virtual void match(int query_id, std::vector<cv::DMatch>& matches) override;

private:
	float m_ratio = 0.6f;
	std::vector<std::vector<cv::DMatch>> m_knn_matches;
	std::shared_ptr<FeaturePool> m_ptr_features;
	std::shared_ptr<cv::FlannBasedMatcher> m_ptr_matcher;
};
