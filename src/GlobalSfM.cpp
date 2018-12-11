#include "GlobalSfM.h"
#include "EpiGraphBuilder.h"
#include "TracksBuilder.h"
#include "TripletsBuilder.h"
#include "Scene.h"
#include "Triangulator.h"
#include "CeresBA.h"
#include "Utils.h"
#include "RejectEdgeByRotation.h"
#include "RejectEdgeByTranslation.h"
#include "RefineRelativeTranslationsWithKnownRotations.h"
#include <boost/graph/graphviz.hpp>

using namespace std;
using namespace Eigen;
using namespace boost;

void GlobalSfM::set_relative_transforms_estimator(std::shared_ptr<RelativeTransformsEstimator> estimator)
{
	m_rel_transforms_est = estimator;
}

void GlobalSfM::set_global_rotations_estimator(std::shared_ptr<GlobalRotationsEstimator> estimator)
{
	m_glb_rotations_est = estimator;
}

void GlobalSfM::set_global_translations_estimator(std::shared_ptr<GlobalTranslationsEstimator> estimator)
{
	m_glb_translations_est = estimator;
}

bool GlobalSfM::reconstruct(std::vector<int> view_ids, std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, ViewCameraBinder & view_camera_binder)
{
	clock_t big_bang = clock();
	EpiGraphBuilder graph_builder;
	graph_builder.build(match_pool, view_ids, options.min_matches_per_edge);
	graph_builder.keep_largest_cc();
	graph_builder.swap(m_graph);

	cout << endl;
	cout << "====================================\n";
	cout << "   Estimating Relative Transforms	 \n";
	cout << "====================================\n";
	clock_t start = clock();
	if (!m_rel_transforms_est->estimate(feature_pool, match_pool, view_camera_binder, m_graph))
	{
		cout << "Estimate relative transforms failed." << endl;
		return false;
	}
	cout << "Cost time: " << (clock() - start) / (double)CLOCKS_PER_SEC << "s" << endl;
	remove_outliers_and_keep_largest_cc(m_graph);
	adjust_transforms_by_indices_order(m_graph);

	cout << endl;
	cout << "====================================\n";
	cout << "       Rejecting Bad Triplets		 \n";
	cout << "====================================\n";
	start = clock();
	if (!reject_edge_by_rotation(m_graph, options.max_rotation_error_in_degree))
	{
		cout << "Reject bad triplets failed." << endl;
		return false;
	}
	cout << "Cost time: " << (clock() - start) / (double)CLOCKS_PER_SEC << "s" << endl;
	remove_outliers_and_keep_largest_cc(m_graph);
	adjust_transforms_by_indices_order(m_graph);

	cout << endl;
	cout << "====================================\n";
	cout << "     Estimating Global Rotations	 \n";
	cout << "====================================\n";
	start = clock();
	if (!m_glb_rotations_est->estimate(feature_pool, match_pool, view_camera_binder, m_graph))
	{
		cout << "Estimate global rotations failed." << endl;
		return false;
	}
	cout << "Cost time: " << (clock() - start) / (double)CLOCKS_PER_SEC << "s" << endl;
	remove_outliers_and_keep_largest_cc(m_graph);
	update_relative_rotations_by_global_rotations(m_graph);
	adjust_transforms_by_indices_order(m_graph);

	cout << endl;
	cout << "====================================\n";
	cout << "     Rejecting Bad Translations	 \n";
	cout << "====================================\n";
	start = clock();
	refine_relative_translations_with_known_rotations(m_graph, feature_pool, match_pool, view_camera_binder);
	reject_edge_by_translation(m_graph, options.translation_projection_tolerance, options.num_project_directions);
	cout << "Cost time: " << (clock() - start) / (double)CLOCKS_PER_SEC << "s" << endl;
	remove_outliers_and_keep_largest_cc(m_graph);
	adjust_transforms_by_indices_order(m_graph);
	
	cout << endl;
	cout << "====================================\n";
	cout << "   Estimating Global Translations	 \n";
	cout << "====================================\n";
	start = clock();
	if (!m_glb_translations_est->estimate(feature_pool, match_pool, view_camera_binder, m_graph))
	{
		cout << "Estimate global translations failed." << endl;
		return false;
	}
	cout << "Cost time: " << (clock() - start) / (double)CLOCKS_PER_SEC << "s" << endl;
	remove_outliers_and_keep_largest_cc(m_graph);
	adjust_transforms_by_indices_order(m_graph);

	cout << endl;
	cout << "====================================\n";
	cout << "           Triangulating            \n";
	cout << "====================================\n";
	m_scene = Scene(feature_pool, match_pool, view_camera_binder, m_graph, 3);
	Triangulator triangulator;
	start = clock();
	triangulator.multi_views_dlt(m_scene, false, options.triangulation_max_reproj_err_in_pixels);
	cout << "Cost time: " << (clock() - start) / (double)CLOCKS_PER_SEC << "s" << endl;
	m_scene.save_to_ply("Init Structure.ply");

	cout << endl;
	cout << "====================================\n";
	cout << "          Bundle Adjustment         \n";
	cout << "====================================\n";
	CeresBA::Options ba_options;
	ba_options.show_ceres_summary = false;
	ba_options.show_verbose = true;
	ba_options.huber_loss_width = options.ba_huber_loss_wdith_in_pixels;
	CeresBA ba(ba_options);
	ba.solve(m_scene, CeresBA::ADJUST_STRUCTURE);

	ba.solve(m_scene, CeresBA::ADJUST_TRANSLATIONS | CeresBA::ADJUST_STRUCTURE);
	ba.solve(m_scene, CeresBA::ADJUST_EXTRINSICS | CeresBA::ADJUST_STRUCTURE);
	ba.solve(m_scene, CeresBA::ADJUST_ALL);

	// retriangulate
	size_t last_num_inliers = 0;
	for (int i = 0; i < options.max_retriangulation_iters; ++i)
	{
		size_t num_inliers = triangulator.multi_views_dlt(m_scene, true, options.retriangelation_max_reproj_err_in_pixels);
		if (num_inliers <= last_num_inliers)
			break;

		last_num_inliers = num_inliers;
		ba.solve(m_scene, CeresBA::ADJUST_ALL);
	}
	cout << "Total time cost: " << (clock() - big_bang) / (double)CLOCKS_PER_SEC << "s" << endl;

	return true;
}

Scene & GlobalSfM::get_scene()
{
	return m_scene;
}

EpipolarGraph & GlobalSfM::get_graph()
{
	return m_graph;
}

//void GlobalSfM::save_graph(std::ofstream & file_stream)
//{
//	vector<string> names;
//	auto vertex_count = num_vertices(m_graph);
//	for (auto i = 0; i < vertex_count; ++i)
//	{
//		names.push_back(std::to_string(m_graph[i].view_id));
//	}
//	write_graphviz(file_stream, m_graph);
//}

void GlobalSfM::remove_outliers_and_keep_largest_cc(EpipolarGraph & epi_graph)
{
	static EpiGraphBuilder graph_builder;
	graph_builder.swap(epi_graph);
	graph_builder.keep_largest_cc();
	graph_builder.swap(epi_graph);
}

void GlobalSfM::update_relative_rotations_by_global_rotations(EpipolarGraph & epi_graph)
{
	EpipolarGraph::edge_iterator e_begin, e_end;
	tie(e_begin, e_end) = edges(epi_graph);
	Matrix3d Ri, Rj, Rij;

	for (auto it_e = e_begin; it_e != e_end; ++it_e)
	{
		size_t i = it_e->m_source;
		size_t j = it_e->m_target;
		if (i > j)	//若 i > j 则交换
		{
			i = i^j;
			j = i^j;
			i = i^j;
		}

		GlobalTransform& transform_i = epi_graph[i];
		GlobalTransform& transform_j = epi_graph[j];
		RelativeTransform& transform_ij = epi_graph[*it_e];

		angle_axis_to_rotation_matrix(transform_i.rt.topRows(3), Ri);
		angle_axis_to_rotation_matrix(transform_j.rt.topRows(3), Rj);

		Rij = Rj * Ri.transpose();

		/*if (transform_i.view_id == transform_ij.src_id)
		{
			Rij = Rj * Ri.transpose();
		}
		else
		{
			Rij = Ri * Rj.transpose();
		}*/
		//cout << "Old relative rotation: " << endl << transform_ij.rt.topRows(3) << endl;
		rotation_matrix_to_angle_axis(Rij, transform_ij.rt.topRows(3));
		//cout << "New relative rotation: " << endl << transform_ij.rt.topRows(3) << endl;
	}
}

void GlobalSfM::adjust_transforms_by_indices_order(EpipolarGraph & epi_graph)
{
	EpipolarGraph::edge_iterator e_begin, e_end;
	tie(e_begin, e_end) = edges(epi_graph);

	for (auto it_e = e_begin; it_e != e_end; ++it_e)
	{
		size_t i = it_e->m_source;
		size_t j = it_e->m_target;
		if (i > j)	//若 i > j 则交换
		{
			i = i^j;
			j = i^j;
			i = i^j;
		}

		auto view_i = epi_graph[i].view_id;
		auto view_j = epi_graph[j].view_id;
		auto& transform = epi_graph[*it_e];
		if (transform.src_id != view_i)
		{
			transform.src_id = view_i;
			transform.dst_id = view_j;
			transform.rt = reversed_transform(transform.rt);
		}
	}
}
