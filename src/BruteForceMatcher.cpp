#include "BruteForceMatcher.h"
#include "MatcherUtils.h"

using namespace std;
using namespace cv;

BruteForceMatcher::BruteForceMatcher(std::shared_ptr<FeaturePool> features, cv::NormTypes norm_type, /*bool cross_check,*/ float ratio)
{
	m_ptr_features = features;
	//OpenCV的ratio test和cross check无法共存的问题
	m_ptr_matcher = make_shared<BFMatcher>(norm_type, false);
	m_ratio = ratio;
}

void BruteForceMatcher::train(int train_id)
{
	auto descriptors_ptr = m_ptr_features->get_descriptors(train_id);
	m_ptr_matcher->clear();
	m_ptr_matcher->add(*descriptors_ptr);
	m_ptr_matcher->train();
}

void BruteForceMatcher::match(int query_id, std::vector<cv::DMatch>& matches)
{
	const auto& query_descriptors = *m_ptr_features->get_descriptors(query_id);
	m_knn_matches.clear();
	m_knn_matches.reserve(query_descriptors.rows);
	m_ptr_matcher->knnMatch(query_descriptors, m_knn_matches, 2);

	ratio_test(m_knn_matches, m_ratio, matches);
}
