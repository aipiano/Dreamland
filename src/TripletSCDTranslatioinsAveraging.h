#pragma once

#include "GlobalTranslationsEstimator.h"
#include "Scene.h"
#include "Triplets.h"
#include <vector>
#include <ceres\ceres.h>

// Average Triplets Translations by Squared Chordal Distance
class TripletSCDTranslationsAveraging final : public GlobalTranslationsEstimator
{
public:
	struct Options
	{

	};

	// Í¨¹ý GlobalTranslationsEstimator ¼Ì³Ð
	virtual bool estimate(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, ViewCameraBinder & view_camera_binder, EpipolarGraph & epi_graph) override;

private:
	void build_scene_from_triplet(
		Triplet & triplet,
		std::shared_ptr<FeaturePool> feature_pool,
		std::shared_ptr<MatchPool> match_pool,
		ViewCameraBinder & view_camera_binder,
		EpipolarGraph & epi_graph,
		Scene & scene
	);

	void init_points3d_randomly(Tracks & tracks);
	void set_ceres_options(ceres::Solver::Options& options);
	double solve_scene(Scene& scene, ceres::Solver::Options& options);
};
