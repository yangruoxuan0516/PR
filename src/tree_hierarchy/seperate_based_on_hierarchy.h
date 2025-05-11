#ifndef SEPERATE_BASED_ON_HIERARCHY_H
#define SEPERATE_BASED_ON_HIERARCHY_H


#include <vector>
#include <Eigen/Core>

std::vector<int> find_subtree(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E, std::vector<std::vector<int>> ancestor_list, int selected_point);
std::vector<int> going_up(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E, std::vector<std::vector<int>> ancestor_list, std::vector<int> current_subtree);

struct Subtree {
    int seed;
    Eigen::RowVector3d color;
    std::vector<int> points;
};


// std::vector<std::vector<int>> seperate_based_on_hierarchy(const Eigen::MatrixXd& V, 
//                                                           const Eigen::MatrixXi& E_mst, 
//                                                           const Eigen::MatrixXd C, 
//                                                           std::vector<Eigen::RowVector3d> type_colors, 
//                                                           std::vector<std::vector<int>> ancestor_list, 
// --- PSEUDO CODE ---
// input: the point cloud V, the edges E_mst, the color C, the type colors, the ancestor list
// output: the division of the point cloud, represented by the color of each point
// algorithm:
// # step 0: Create a subtree structure to store the subtrees: subtree = {selected_point_of_the_subtree, color_of_the_subtree, points_in_the_subtree}.
// # step 1: Get the raw subtree for each colored point, with find_subtree function, and store the result in a list of subtree structures raw_subtree_list.
// # step 2: For two subtrees st1 and st2, if (a) they have the same color (b) the points in st1 are all in st2, then merge them, that is to say, remove st1 from the list. Finally, we get a list of subtrees subtree_list.
// # step 3: move up all subtrees in the subtree_list with the going_up function, and store the result in a new list of subtrees new_subtree_list.
//           if there is no overlap between the new subtrees, then we can keep them.
//           however, if there is an overlap, between two new subtrees nst1 and nst2, (1) if they are of the same level (same number of ancestors), then we remove the going_up and keep them at the two sides of a split point.
//                                                                                    (2) if they are not of the same level, then we keep the one with a higher level (fewer ancestors) as the new subtree to participate in the next going_up, however, we keep the color of the lower level one.
// we do this until no subtree can be moved up or ...           


Eigen::MatrixXd seperate_based_on_hierarchy(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E_mst,
    const Eigen::MatrixXd& C,
    const std::vector<Eigen::RowVector3d>& type_colors,
    const std::vector<std::vector<int>>& ancestor_list);

#endif // SEPERATE_BASED_ON_HIERARCHY_H