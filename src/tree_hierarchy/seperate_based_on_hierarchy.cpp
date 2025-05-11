#include "seperate_based_on_hierarchy.h"
#include <iostream>

std::vector<int> find_subtree(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E, std::vector<std::vector<int>> ancestor_list, int selected_point) {

    std::vector<int> subtree;
    // first, go up and down until the first split points
    std::vector<int> degree(V.rows(), 0);

    int up, down;

    for (int i = 0; i < E.rows(); ++i) {
        degree[E(i, 0)]++;
        degree[E(i, 1)]++;
    }
    int idx = selected_point;
    std::vector<bool> visited(V.rows(), false);
    visited[idx] = true;
    while(degree[idx] == 2) {
        for (int i = 0; i < E.rows(); ++i) {
            if ( (E(i, 0) == idx && !visited[E(i, 1)]) || (E(i, 1) == idx && !visited[E(i, 0)]) ) {
                idx = E(i, 0) == idx ? E(i, 1) : E(i, 0);
                visited[idx] = true;
                subtree.push_back(idx);
                break;
            }
        }
    }
    if (ancestor_list[idx].size() < ancestor_list[selected_point].size()) {
        up = idx;
    } 
    else {
        down = idx;
    }
    // then go from the other side
    idx = selected_point;
    while(degree[idx] == 2) {
        for (int i = 0; i < E.rows(); ++i) {
            if ( (E(i, 0) == idx && !visited[E(i, 1)]) || (E(i, 1) == idx && !visited[E(i, 0)]) ) {
                idx = E(i, 0) == idx ? E(i, 1) : E(i, 0);
                visited[idx] = true;
                subtree.push_back(idx);
                break;
            }
        }
    }
    if (ancestor_list[idx].size() < ancestor_list[selected_point].size()) {
        up = idx;
    } 
    else {
        down = idx;
    }

    // TODO: i haven't thought about the case when the selected point is not in the middle of a branch
    // what's more, now if a branch is colorer, we cannot select a subtree from it for a new label, because it goes back to grey

    std::vector<int> selected_ancestors = ancestor_list[selected_point];
    selected_ancestors.push_back(down);
    int ancestor_list_length = selected_ancestors.size();

    // for all points, if the beginning of the ancestor list is the same, then they are siblings / descendants
    for (int i = 0; i < ancestor_list.size(); ++i) {
        if (i == selected_point) continue;
        if (ancestor_list[i].size() < ancestor_list_length) continue; 
        // compare the first ancestor_list_length elements
        bool is_subtree = true;
        for (int j = 0; j < ancestor_list_length; ++j) {
            if (ancestor_list[i][j] != selected_ancestors[j]) {
                is_subtree = false;
                break;
            }
        }
        if (is_subtree) {
            subtree.push_back(i);
        }
    }

    return subtree;
}


std::vector<int> going_up(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E, std::vector<std::vector<int>> ancestor_list, std::vector<int> current_subtree){
    // first, find the point with the least ancestors
    int min_ancestors = ancestor_list[current_subtree[0]].size();
    int min_index = current_subtree[0];
    for (int i = 1; i < current_subtree.size(); ++i) {
        if (ancestor_list[current_subtree[i]].size() < min_ancestors) {
            min_ancestors = ancestor_list[current_subtree[i]].size();
            min_index = current_subtree[i];
        }
    }

    // then, get its ancestor list
    std::vector<int> common_ancestor = ancestor_list[min_index];
    // add this point to the common ancestor
    common_ancestor.push_back(min_index);

    // then, find all the points that have this common ancestor
    std::vector<int> new_subtree;
    for (int i = 0; i < ancestor_list.size(); ++i) {
        if (i == current_subtree[0]) continue;
        if (ancestor_list[i].size() < common_ancestor.size()) continue; 
        // compare the first common_ancestor.size() elements
        bool is_subtree = true;
        for (int j = 0; j < common_ancestor.size(); ++j) {
            if (ancestor_list[i][j] != common_ancestor[j]) {
                is_subtree = false;
                break;
            }
        }
        if (is_subtree) {
            new_subtree.push_back(i);
        }
    }

    // add this point to the new subtree
    new_subtree.push_back(min_index);

    // going up till the first split point
    std::vector<int> degree(V.rows(), 0);
    for (int i = 0; i < E.rows(); ++i) {
        degree[E(i, 0)]++;
        degree[E(i, 1)]++;
    }
    int idx = min_index;
    std::vector<bool> visited(V.rows(), false);
    visited[idx] = true;
    // mark the points in the new subtree as visited
    for (int i = 0; i < new_subtree.size(); ++i) {
        visited[new_subtree[i]] = true;
    }
    for (int i = 0; i < E.rows(); ++i) {
        if ( (E(i, 0) == idx && !visited[E(i, 1)]) || (E(i, 1) == idx && !visited[E(i, 0)]) ) {
            idx = E(i, 0) == idx ? E(i, 1) : E(i, 0);
            visited[idx] = true;
            new_subtree.push_back(idx);
            break;
        }
    }
    while(degree[idx] == 2) {
        for (int i = 0; i < E.rows(); ++i) {
            if ( (E(i, 0) == idx && !visited[E(i, 1)]) || (E(i, 1) == idx && !visited[E(i, 0)]) ) {
                idx = E(i, 0) == idx ? E(i, 1) : E(i, 0);
                visited[idx] = true;
                new_subtree.push_back(idx);
                break;
            }
        }
    }

    return new_subtree;
}





Eigen::MatrixXd seperate_based_on_hierarchy(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E_mst,
    const Eigen::MatrixXd& C,
    const std::vector<Eigen::RowVector3d>& type_colors,
    const std::vector<std::vector<int>>& ancestor_list)
{
    using namespace std;

    std::vector<Subtree> raw_subtree_list;

    // STEP 0 & 1: initialize raw subtrees from colored points
    for (int i = 0; i < V.rows(); ++i) {
        Eigen::RowVector3d color = C.row(i);
        for (const auto& type_color : type_colors) {
            if ((color - type_color).norm() < 1e-6) {
                Subtree st;
                st.seed = i;
                st.color = type_color;
                st.points = find_subtree(V, E_mst, ancestor_list, i);
                st.points.push_back(i);
                std::sort(st.points.begin(), st.points.end());  // ensure sorted for later
                raw_subtree_list.push_back(st);
                break;
            }
        }
    }

    // STEP 2: merge nested subtrees
    std::vector<Subtree> subtree_list;
    for (int i = 0; i < raw_subtree_list.size(); ++i) {
        bool should_add = true;
        for (int j = 0; j < raw_subtree_list.size(); ++j) {
            if (i == j) continue;
            if ((raw_subtree_list[i].color - raw_subtree_list[j].color).norm() < 1e-6) {
                const auto& A = raw_subtree_list[i].points;
                const auto& B = raw_subtree_list[j].points;
                if (std::includes(B.begin(), B.end(), A.begin(), A.end())) {
                    should_add = false;
                    break;
                }
            }
        }
        if (should_add) {
            subtree_list.push_back(raw_subtree_list[i]);
        }
    }

    // STEP 3: iterative going up
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<Subtree> new_subtree_list;

        for (auto& st : subtree_list) {
            Subtree new_st;
            new_st.seed = st.seed;
            new_st.color = st.color;
            new_st.points = going_up(V, E_mst, ancestor_list, st.points);
            std::sort(new_st.points.begin(), new_st.points.end());
            new_subtree_list.push_back(new_st);
        }

        std::vector<bool> keep(new_subtree_list.size(), true);
        for (int i = 0; i < new_subtree_list.size(); ++i) {
            for (int j = i + 1; j < new_subtree_list.size(); ++j) {
                std::vector<int> intersect;
                std::set_intersection(
                    new_subtree_list[i].points.begin(), new_subtree_list[i].points.end(),
                    new_subtree_list[j].points.begin(), new_subtree_list[j].points.end(),
                    std::back_inserter(intersect));

                if (!intersect.empty()) {
                    int l1 = ancestor_list[new_subtree_list[i].seed].size();
                    int l2 = ancestor_list[new_subtree_list[j].seed].size();
                    if (l1 == l2) {
                        keep[i] = false;
                        keep[j] = false;
                    } else {
                        int higher = (l1 < l2) ? i : j;
                        int lower = (l1 < l2) ? j : i;
                        keep[lower] = false;
                        changed = true;
                    }
                }
            }
        }

        subtree_list.clear();
        for (int i = 0; i < new_subtree_list.size(); ++i) {
            if (keep[i]) {
                subtree_list.push_back(new_subtree_list[i]);
            }
        }
    }

    // Final color matrix (Eigen::MatrixXd)
    Eigen::MatrixXd final_colors(V.rows(), 3);
    final_colors.setConstant(0.5);  // default gray

    for (const auto& st : subtree_list) {
        for (int idx : st.points) {
            final_colors.row(idx) = st.color;
        }
    }

    return final_colors;
}