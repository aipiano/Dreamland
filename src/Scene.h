#pragma once

#include "Tracks.h"
#include "GlobalTransform.h"
#include "EpipolarGraph.h"
#include "ViewCameraBinder.h"
#include "FeaturePool.h"
#include "MatchPool.h"
#include <Eigen\Eigen>
#include <map>

class Scene
{
public:
	Scene();
	Scene(
		std::shared_ptr<FeaturePool> feature_pool,
		std::shared_ptr<MatchPool> match_pool,
		ViewCameraBinder& view_camera_binder,
		EpipolarGraph& epi_graph,
		int min_track_length = 2
	);

	void export_points(std::vector<Eigen::Vector3d>& points);
	void save_to_ply(std::string name);
	void initialize(
		std::shared_ptr<FeaturePool> feature_pool,
		std::shared_ptr<MatchPool> match_pool,
		ViewCameraBinder& view_camera_binder,
		EpipolarGraph& epi_graph,
		int min_track_length = 2
	);

public:
	//set by user
	Tracks tracks;
	//set by user
	std::vector<GlobalTransform> views;
	//set by user
	ViewCameraBinder view_camera_binder;
	//set by user. get view index in member views by view id.
	std::map<int, int> view_idx_by_view_id;
};
