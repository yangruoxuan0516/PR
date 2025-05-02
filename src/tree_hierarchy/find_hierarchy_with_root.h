#ifndef FIND_HIERARCHY_WITH_ROOT_H
#define FIND_HIERARCHY_WITH_ROOT_H

#include <Eigen/Core>
#include <igl/opengl/glfw/Viewer.h>

int find_root(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E);

#endif // FIND_HIERARCHY_WITH_ROOT_H