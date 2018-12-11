#include "MatchFrame.h"

MatchFrame::MatchFrame()
{
}

MatchFrame::MatchFrame(const MatchFrame & other)
{
	train_id = other.train_id;
	query_id = other.query_id;
	matches = other.matches;
	//mask = other.mask;
}

MatchFrame::MatchFrame(MatchFrame && other)
{
	train_id = other.train_id;
	query_id = other.query_id;
	matches = std::move(other.matches);
	//mask = std::move(other.mask);
}

MatchFrame::MatchFrame(const std::vector<cv::DMatch>& matches, int train_id, int query_id /*, cv::Mat& mask*/)
	: train_id(train_id), query_id(query_id), matches(matches)//, mask(mask)
{
}

MatchFrame::MatchFrame(std::vector<cv::DMatch>&& matches, int train_id, int query_id /*, cv::Mat& mask*/)
	: train_id(train_id), query_id(query_id), matches(std::move(matches))//, mask(mask)
{
}

MatchFrame & MatchFrame::operator=(const MatchFrame & other)
{
	train_id = other.train_id;
	query_id = other.query_id;
	matches = other.matches;
	//mask = other.mask;
	return *this;
}

void MatchFrame::operator=(MatchFrame && other)
{
	train_id = other.train_id;
	query_id = other.query_id;
	matches = std::move(other.matches);
	//mask = std::move(other.mask);
}
