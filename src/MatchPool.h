#pragma once

#include <opencv2\features2d.hpp>
#include <vector>
#include <memory>
#include "PairwiseMatch.h"
#include "MatchFrame.h"

class MatchPool
{
public:
	typedef std::shared_ptr<const MatchFrame> MatchFramePtr;
	virtual void push(const std::vector<cv::DMatch>& matches, int train_id, int query_id) = 0;
	virtual void push(std::vector<cv::DMatch>&& matches, int train_id, int query_id) = 0;
	//virtual void set_mask(const cv::Mat& mask, int train_id, int query_id) = 0;
	// 返回指针而不是引用，使得以后可以支持IO cache
	virtual const MatchFramePtr get_matches(const PairwiseMatch& match_pair) = 0;
	virtual const MatchFramePtr get_matches(int view_id1, int view_id2) = 0;
	virtual size_t size() = 0;
	virtual void clear() = 0;
};
