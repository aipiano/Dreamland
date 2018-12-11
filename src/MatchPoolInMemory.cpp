#include "MatchPoolInMemory.h"

using namespace std;
using namespace cv;

void MatchPoolInMemory::push(const vector<DMatch>& matches, int train_id, int query_id)
{
	MatchFramePtr frame = make_shared<MatchFrame>(matches, train_id, query_id);
	push(frame);
}

void MatchPoolInMemory::push(vector<DMatch>&& matches, int train_id, int query_id)
{
	MatchFramePtr frame = make_shared<MatchFrame>(std::move(matches), train_id, query_id);
	push(frame);
}

void MatchPoolInMemory::push(const MatchFramePtr match_frame)
{
	//保证匹配的无向性
	PairwiseMatch pair;
	pair.id1 = min(match_frame->train_id, match_frame->query_id);
	pair.id2 = max(match_frame->train_id, match_frame->query_id);

	m_mutex.lock();
	m_matches_map[pair] = match_frame;
	m_mutex.unlock();
}

//void MatchPoolInMemory::set_mask(const cv::Mat & mask, int train_id, int query_id)
//{
//	//保证匹配的无向性
//	PairwiseMatch pair;
//	pair.id1 = min(train_id, query_id);
//	pair.id2 = max(train_id, query_id);
//
//	m_mutex.lock();
//	const auto* frame_ptr = m_matches_map[pair].get();
//	if (frame_ptr == nullptr)
//	{
//		m_mutex.unlock();
//		return;
//	}
//	const_cast<MatchFrame*>(frame_ptr)->mask = mask.clone();
//	m_mutex.unlock();
//}

//void MatchPoolInMemory::push(const MatchFrame& match_frame)
//{
//	//保证匹配的无向性
//	PairwiseMatch pair;
//	pair.id1 = min(match_frame.train_id, match_frame.query_id);
//	pair.id2 = max(match_frame.train_id, match_frame.query_id);
//
//	m_mutex.lock();
//	m_matches_map[pair] = match_frame;
//	m_mutex.unlock();
//}
//
//void MatchPoolInMemory::push(MatchFrame&& match_frame)
//{
//	//保证匹配的无向性
//	PairwiseMatch pair;
//	pair.id1 = min(match_frame.train_id, match_frame.query_id);
//	pair.id2 = max(match_frame.train_id, match_frame.query_id);
//
//	m_mutex.lock();
//	m_matches_map[pair] = std::move(match_frame);
//	m_mutex.unlock();
//}

const MatchPoolInMemory::MatchFramePtr MatchPoolInMemory::get_matches(const PairwiseMatch& match_pair)
{
	//保证匹配的无向性
	PairwiseMatch pair;
	pair.id1 = min(match_pair.id1, match_pair.id2);
	pair.id2 = max(match_pair.id1, match_pair.id2);

	auto find_result = m_matches_map.find(pair);
	if (find_result == m_matches_map.end())
	{
		return nullptr;
	}
	return find_result->second;
}

const MatchPoolInMemory::MatchFramePtr MatchPoolInMemory::get_matches(int view_id1, int view_id2)
{
	return get_matches(PairwiseMatch(view_id1, view_id2));
}

size_t MatchPoolInMemory::size()
{
	return m_matches_map.size();
}

void MatchPoolInMemory::clear()
{
	m_mutex.lock();
	m_matches_map.clear();
	m_mutex.unlock();
}
