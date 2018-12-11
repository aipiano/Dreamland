#pragma once

#include <Eigen/Eigen>

//pinhole: fx, fy, cx, cy, k1, k2, p1, p2, k3;
//Taylor model: a0, a2, a3, a4, cx, cy, a, b, c;
typedef Eigen::VectorXd CameraIntrinsic;
