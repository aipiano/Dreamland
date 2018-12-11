#pragma once

#include <opencv2\features2d.hpp>
#include <vector>
#include <map>
#include <mutex>

#include "PairwiseMatch.h"
#include "MatchFrame.h"
#include "MatchPool.h"

class MatchPoolInMemory final : public MatchPool
{
public:
	MatchPoolInMemory() {};
	virtual void push(const std::vector<cv::DMatch>& matches, int train_id, int query_id) override;
	virtual void push(std::vector<cv::DMatch>&& matches, int train_id, int query_id) override;
	//virtual void set_mask(const cv::Mat& mask, int train_id, int query_id) override;
	virtual const MatchFramePtr get_matches(const PairwiseMatch& match_pair) override;
	virtual const MatchFramePtr get_matches(int view_id1, int view_id2) override;
	virtual size_t size() override;
	virtual void clear() override;

private:
	void push(const MatchFramePtr match_frame);

private:
	typedef std::map<PairwiseMatch, MatchFramePtr> MatchesMap;

	std::mutex m_mutex;
	MatchesMap m_matches_map;
	//const MatchFrame m_matches_empty;
};