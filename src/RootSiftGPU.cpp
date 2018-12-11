#include "RootSiftGPU.h"

#include <string>
#include <iostream>

using namespace std;
using namespace cv;

#define GL_LUMINANCE 0x1909
#define GL_RGB 0x1907
#define GL_UNSIGNED_BYTE 0x1401
#define GL_FLOAT 0x1406

RootSiftGPU::RootSiftGPU(int first_octave, int num_dog_levels, float dog_thresh, float edge_thresh, int device_id)
{
	init_sift_gpu(first_octave, num_dog_levels, dog_thresh, edge_thresh, device_id);
}

RootSiftGPU::~RootSiftGPU()
{
	free_sift_gpu();
}

bool RootSiftGPU::extract(cv::Mat & image, std::vector<cv::KeyPoint>& keypoints, cv::Mat & descriptors)
{
	if (image.empty()) return false;

	switch (image.type())
	{
	case CV_8UC1:
		m_sift->RunSIFT(image.cols, image.rows, image.data, GL_LUMINANCE, GL_UNSIGNED_BYTE);
		break;
	case CV_8UC3:
		m_sift->RunSIFT(image.cols, image.rows, image.data, GL_RGB, GL_UNSIGNED_BYTE);
		break;
	default:
		cout << "Unsupport image format." << endl;
		return false;
	}

	int num = m_sift->GetFeatureNum();
	m_sift_keypoints_buf.resize(num);

	keypoints.resize(num);
	descriptors.create(num, 128, CV_32FC1);
	m_sift->GetFeatureVector(m_sift_keypoints_buf.data(), descriptors.ptr<float>());

	for (int i = 0; i < num; ++i)
	{
		KeyPoint& kp = keypoints[i];
		SiftKeypoint& sift_kp = m_sift_keypoints_buf[i];
		kp.pt.x = sift_kp.x;
		kp.pt.y = sift_kp.y;
		kp.size = sift_kp.s;
		kp.angle = sift_kp.o;

		//Convert to RootSIFT Descriptors
		normalize(descriptors.row(i), descriptors.row(i), 1, 0, NORM_L1);
	}
	sqrt(descriptors, descriptors);

	return true;
}

bool RootSiftGPU::init_sift_gpu(int first_octave, int num_dog_levels, float dog_thresh, float edge_thresh, int device_id)
{
	m_sift = new SiftGPU();

	if (!m_sift)
		return false;

	string str_fo = to_string(first_octave);
	string str_dn = to_string(num_dog_levels);
	string str_dt = to_string(dog_thresh);
	string str_et = to_string(edge_thresh);
	string str_device_id = to_string(device_id);
	char * argv[] = {
		"-fo", (char*)str_fo.c_str(),
		"-d", (char*)str_dn.c_str(),
		"-t", (char*)str_dt.c_str(),
		"-e", (char*)str_et.c_str(),
		"-cuda", (char*)str_device_id.c_str(),
		"-v", "0", 
		"-tc2", "6000", 
		"-unn", 
	};
	//-fo -1    staring from -1 octave 
	//-v 0      no output at all, except errors
	//-loweo    add a (.5, .5) offset
	//-tc <num> set a soft limit to number of detected features
	//-unn		write unnormalized descriptors

	//NEW:  parameters for  GPU-selection
	//1. CUDA.   Use parameter "-cuda", "[device_id]"
	//2. OpenGL. Use "-Display", "display_name" to select monitor/GPU (XLIB/GLUT)
	//   		 on windows the display name would be something like \\.\DISPLAY4

	int argc = sizeof(argv) / sizeof(char*);
	m_sift->ParseParam(argc, argv);
	//Create a context for computation, and SiftGPU will be initialized automatically 
	//The same context can be used by SiftMatchGPU
	if (m_sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
		return false;

	return true;
}

void RootSiftGPU::free_sift_gpu()
{
	if (m_sift)
	{
		delete m_sift;
		m_sift = nullptr;
	}
}
