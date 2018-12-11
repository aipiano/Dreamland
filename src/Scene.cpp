#include "Scene.h"
#include "TracksBuilder.h"
#include "Utils.h"
#include <fstream>

using namespace std;
using namespace Eigen;

Scene::Scene()
{
}

Scene::Scene(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, ViewCameraBinder & view_camera_binder, EpipolarGraph & epi_graph, int min_track_length)
{
	initialize(feature_pool, match_pool, view_camera_binder, epi_graph, min_track_length);
}

void Scene::export_points(std::vector<Eigen::Vector3d>& points)
{
	points.clear();
	for (auto& track : tracks)
	{
		if (!track.is_inlier) continue;
		points.push_back(track.point3d);
	}
}

void Scene::save_to_ply(std::string name)
{
	vector<Vector3d> points;
	export_points(points);

	ofstream ofs(name);
	// write header
	ofs << "ply" << endl;
	ofs << "format ascii 1.0" << endl;
	ofs << "comment made by Dreamland" << endl;
	ofs << "element vertex " << points.size() + views.size() << endl;
	ofs << "property float x" << endl;
	ofs << "property float y" << endl;
	ofs << "property float z" << endl;
	ofs << "property uchar red" << endl;
	ofs << "property uchar green" << endl;
	ofs << "property uchar blue" << endl;
	//ofs << "element face " << triangleNum << endl;	// write points only for now!
	//ofs << "property list uchar int vertex_index" << endl;
	ofs << "end_header" << endl;

	// Save point cloud
	for (auto&p : points)
	{
		ofs << (float)p[0] << " " << (float)p[1] << " " << (float)p[2] << " 255 255 255" << endl;
	}

	// Save camera positions
	Matrix3d R;
	for (auto& view : views)
	{
		//angle_axis_to_rotation_matrix(view.rt.topRows(3), R);
		Vector3d p = extrinsic_to_position(view.rt);
		ofs << (float)p[0] << " " << (float)p[1] << " " << (float)p[2] << " 255 0 0" << endl;
	}
	ofs.close();
}

void Scene::initialize(
	std::shared_ptr<FeaturePool> feature_pool, 
	std::shared_ptr<MatchPool> match_pool,
	ViewCameraBinder & view_camera_binder, 
	EpipolarGraph & epi_graph,
	int min_track_length
)
{
	this->tracks.clear();
	this->views.clear();
	this->view_camera_binder.clear();
	this->view_idx_by_view_id.clear();

	TracksBuilder tracks_builder;
	tracks_builder.build(feature_pool, match_pool, epi_graph);
	tracks_builder.filter(min_track_length);
	tracks_builder.swap(this->tracks);

	EpipolarGraph::vertex_iterator v_begin, v_end;
	tie(v_begin, v_end) = vertices(epi_graph);
	for (auto v_it = v_begin; v_it != v_end; ++v_it)
	{
		auto& view = epi_graph[*v_it];
		if (!view.is_inlier) continue;

		this->views.push_back(view);
		this->view_idx_by_view_id[view.view_id] = this->views.size() - 1;
		this->view_camera_binder.bind(view.view_id, view_camera_binder.get_camera(view.view_id));
	}
}
