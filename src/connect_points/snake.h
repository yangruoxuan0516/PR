#ifndef SNAKE_H
#define SNAKE_H

#include <Eigen/Core>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <params.h>

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> snake(GUIParams& params, const Eigen::MatrixXd& points, const Eigen::RowVectorXd start_point, const Eigen::RowVectorXd end_point);

#endif