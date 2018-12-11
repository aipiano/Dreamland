#pragma once

#include "RelativeTransformsEstimator.h"
#include "GlobalRotationsEstimator.h"
#include "GlobalTranslationsEstimator.h"
#include "FeaturePool.h"
#include "MatchPool.h"
#include "ViewCameraBinder.h"
#include "Triplets.h"
#include "EpipolarGraph.h"
#include "Scene.h"

#include <memory>
#include <fstream>

class GlobalSfM
{
public:
	struct Options
	{
		size_t min_matches_per_edge = 50;
		double max_rotation_error_in_degree = 5;
		double translation_projection_tolerance = 0.01;
		size_t num_project_directions = 50;		// this should be larger than 40 for recommandation
		double ba_huber_loss_wdith_in_pixels = 16;
		double triangulation_max_reproj_err_in_pixels = 8;
		double retriangelation_max_reproj_err_in_pixels = 4;
		size_t max_retriangulation_iters = 5;
	} options;

	void set_relative_transforms_estimator(std::shared_ptr<RelativeTransformsEstimator> estimator);
	void set_global_rotations_estimator(std::shared_ptr<GlobalRotationsEstimator> estimator);
	void set_global_translations_estimator(std::shared_ptr<GlobalTranslationsEstimator> estimator);

	bool reconstruct(std::vector<int> view_ids, std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, ViewCameraBinder& view_camera_binder);
	Scene& get_scene();
	EpipolarGraph& get_graph();
	//void save_graph(std::ofstream& file_stream);

private:
	void remove_outliers_and_keep_largest_cc(EpipolarGraph& epi_graph);
	void update_relative_rotations_by_global_rotations(EpipolarGraph& epi_graph);
	// 调整变换方向，使所有变换表示为从顶点i到顶点j的变换，且i<j
	void adjust_transforms_by_indices_order(EpipolarGraph& epi_graph);

private:
	std::shared_ptr<RelativeTransformsEstimator> m_rel_transforms_est;
	std::shared_ptr<GlobalRotationsEstimator> m_glb_rotations_est;
	std::shared_ptr<GlobalTranslationsEstimator> m_glb_translations_est;
	Scene m_scene;
	EpipolarGraph m_graph;
};
