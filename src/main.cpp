#include "RootSiftGPU.h"
#include "FeaturePoolInMemory.h"
#include "MatchPoolInMemory.h"
#include "ViewCameraBinder.h"
#include "BruteForceMatcher.h"
#include "tinydir.h"
#include "FivePointsEstimator.h"
#include "Pinhole.hpp"
#include "FlannMatcher.h"
#include "LinearRotationsAveraging.h"
#include "TrilinearTranslationsAveraging.h"
#include "RobustRotationsAveraging.h"
#include "GlobalSfM.h"
#include "CasHashMatcher.h"
#include "LinearTranslationsAveraging.h"

#include <string>
#include <vector>
#include <opencv2\highgui.hpp>
#include <memory>
#include <opencv2\imgproc.hpp>
#include <iostream>
#include <random>
#include <string>

using namespace std;
using namespace cv;
using namespace boost;
using namespace Eigen;

void get_file_names(string dir_name, vector<string> & names)
{
	names.clear();
	tinydir_dir dir;
	tinydir_open(&dir, dir_name.c_str());

	while (dir.has_next)
	{
		tinydir_file file;
		tinydir_readfile(&dir, &file);
		if (!file.is_dir)
		{
			names.push_back(file.path);
		}
		tinydir_next(&dir);
	}
	tinydir_close(&dir);
}

void main()
{
	/*MatrixXd A(100, 100);
	A.setRandom();

	MatrixXd K(100, 100);
	K.setRandom();

	MatrixXd B(200, 200);

	clock_t start, stop;
	double delta;

	Vector3d a;
	a.setRandom();

	start = clock();
	B.block(0, 0, 100, 100).triangularView<Upper>() = a.squaredNorm() * A.transpose() * A;
	B.block(0, 0, 100, 100).triangularView<Upper>() += 4 * K.transpose() * K;
	stop = clock();
	delta = (stop - start);
	cout << delta << endl;

	start = clock();
	B.block(0, 0, 100, 100) = a.squaredNorm() * A.transpose() * A + 4 * K.transpose() * K;
	stop = clock();
	delta = (stop - start);
	cout << delta << endl;*/

	vector<Mat> images;
	vector<string> image_names;
	get_file_names("D:\\MyProjects\\Dreamland_v0.9\\data\\castle_dense_large", image_names);
	//get_file_names("D:\\MyProjects\\Dreamland_v0.9\\data\\herzjesu_dense_large", image_names);
	//get_file_names("D:\\MyProjects\\Dreamland_v0.9\\data\\fountain_dense", image_names);
	//get_file_names("D:\\MyProjects\\Dreamland\\data\\mechanic1", image_names);
	
	RootSiftGPU siftgpu;
	std::shared_ptr<FeaturePool> feature_pool = make_shared<FeaturePoolInMemory>();
	ViewCameraBinder view_camera_binder;

	Eigen::Vector4d intrinsic;
	intrinsic << 2759.48, 2764.16, 1536, 1024;

	std::shared_ptr<Camera> pinhole = make_shared<Pinhole>(intrinsic);
	
	int view_id = 0;
	vector<int> view_ids/*{ 3, 1, 2 }*/;
	//std::random_shuffle(view_ids.begin(), view_ids.end());
	for (auto& name : image_names)
	{
		vector<KeyPoint> key_points;
		Mat descriptors;

		images.push_back(imread(name));
		cout << "Extract features from image: " << name << endl;

		siftgpu.extract(images.back(), key_points, descriptors);
		feature_pool->push(std::move(key_points), descriptors, view_id);
		view_ids.push_back(view_id);

		view_camera_binder.bind(view_id, pinhole);

		++view_id;
	}
	
	std::shared_ptr<MatchPool> match_pool = make_shared<MatchPoolInMemory>();
	BruteForceMatcher bf_matcher(feature_pool);
	FlannMatcher flann_matcher(feature_pool, 0.6f);
	CasHashMatcher cashash_matcher(feature_pool, 6, 10, 0.6f);
	
	for (int i = 0; i < view_ids.size(); ++i)
	{
		int train_id = view_ids[i];
		//bf_matcher.train(train_id);
		//flann_matcher.train(train_id);
		cashash_matcher.train(train_id);
		int upper = view_ids.size();
		//int upper = min(i + 5, view_ids.size());
		for (int j = i + 1; j < upper; ++j)
		{
			cout << "Matching pair (" << view_ids[i] << ", " << view_ids[j] << ")" << endl;
			int query_id = view_ids[j];
			vector<DMatch> matches;
			//bf_matcher.match(query_id, matches);
			//flann_matcher.match(query_id, matches);
			cashash_matcher.match(query_id, matches);

			match_pool->push(std::move(matches), train_id, query_id);
		}
	}
	
	FivePointsEstimator::Options fv_options;
	fv_options.min_num_inliers = 30;
	std::shared_ptr<RelativeTransformsEstimator> five_point_est = make_shared<FivePointsEstimator>(fv_options);

	//std::shared_ptr<GlobalRotationsEstimator> rotations_avg = make_shared<LinearRotationsAveraging>();
	RobustRotationsAveraging::Options rav_options;
	std::shared_ptr<GlobalRotationsEstimator> rotations_avg = make_shared<RobustRotationsAveraging>(rav_options);

	TrilinearTranslationsAveraging::Options trifocal_options;
	trifocal_options.min_num_inliers = 30;
	std::shared_ptr<GlobalTranslationsEstimator> translations_avg = make_shared<TrilinearTranslationsAveraging>(trifocal_options);

	LinearTranslationsAveraging::Options linear_options;
	//std::shared_ptr<GlobalTranslationsEstimator> translations_avg = make_shared<LinearTranslationsAveraging>(linear_options);

	GlobalSfM sfm;
	sfm.set_relative_transforms_estimator(five_point_est);
	sfm.set_global_rotations_estimator(rotations_avg);
	sfm.set_global_translations_estimator(translations_avg);

	sfm.reconstruct(view_ids, feature_pool, match_pool, view_camera_binder);
	Scene& scene = sfm.get_scene();
	scene.save_to_ply("D3D.ply");

	/*ofstream ofs("graph.dot");
	sfm.save_graph(ofs);*/
}