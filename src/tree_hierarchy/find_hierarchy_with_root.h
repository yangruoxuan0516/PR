#ifndef FIND_HIERARCHY_WITH_ROOT_H
#define FIND_HIERARCHY_WITH_ROOT_H

#include <Eigen/Core>
#include <igl/opengl/glfw/Viewer.h>

int find_root(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E);

// assign to each point, a list of ancestors
// so that we can know - its level and - its ancestors
// then if two points have the same ancestor list, they are assign to the same "subtree"

std::vector<std::vector<int>> find_ancestor_list(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E, int root);


// then cf the paper

#endif // FIND_HIERARCHY_WITH_ROOT_H