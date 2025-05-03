#include "find_hierarchy_with_root.h"

int find_root(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E) {
    int root;

    // find all the ending points in the tree
    std::vector<int> end_points;
    std::vector<int> degree(V.rows(), 0);
    for (int i = 0; i < E.rows(); ++i) {
        degree[E(i, 0)]++;
        degree[E(i, 1)]++;
    }

    for (int i = 0; i < degree.size(); ++i) {
        if (degree[i] == 1) {
            end_points.push_back(i);
        }
    }

    // calculate the distance from each end point to its first split, by adding the distance to the root
    std::vector<double> distances(end_points.size(), 0);
    for (int i = 0; i < end_points.size(); ++i) {
        int idx = end_points[i];

        // create a list to store the visited points
        std::vector<bool> visited(V.rows(), false);
        visited[idx] = true;

        // find the edge that connects to this end point
        for (int j = 0; j < E.rows(); ++j) {
            if (E(j, 0) == idx || E(j, 1) == idx) {
                visited[E(j, 0)] = true;
                visited[E(j, 1)] = true;
                idx = E(j, 0) == idx ? E(j, 1) : E(j, 0);
                break;
            }
        }
        // add the distance to the root
        distances[i] = (V.row(end_points[i]) - V.row(idx)).norm();
        // find the next edges
        while (degree[idx] == 2) {
            for (int j = 0; j < E.rows(); ++j) {
                if ( (E(j, 0) == idx && !visited[E(j, 1)]) || (E(j, 1) == idx && !visited[E(j, 0)]) ) {
                    visited[E(j, 0)] = true;
                    visited[E(j, 1)] = true;
                    idx = E(j, 0) == idx ? E(j, 1) : E(j, 0);
                    break;
                }
            }
            distances[i] += (V.row(end_points[i]) - V.row(idx)).norm();
        }
    }

    // find the end point with the maximum distance
    double max_distance = 0;
    for (int i = 0; i < distances.size(); ++i) {
        if (distances[i] > max_distance) {
            max_distance = distances[i];
            root = end_points[i];
        }
    }

    return root;
}



std::vector<std::vector<int>> find_ancestor_list(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E, int root) {
    // Step 1: Build adjacency list
    std::vector<std::vector<int>> adj(V.rows());
    for (int i = 0; i < E.rows(); ++i) {
        adj[E(i, 0)].push_back(E(i, 1));
        adj[E(i, 1)].push_back(E(i, 0));
    }

    std::vector<std::vector<int>> ancestor_list(V.rows());
    std::vector<bool> visited(V.rows(), false);

    std::function<void(int, int, std::vector<int>)> dfs = [&](int node, int parent, std::vector<int> ancestors) {
        visited[node] = true;
        ancestor_list[node] = ancestors;

        // Count children (excluding parent)
        int child_count = 0;
        for (int neighbor : adj[node]) {
            if (neighbor != parent && !visited[neighbor]) {
                child_count++;
            }
        }

        // If split (i.e., more than one child), add self to ancestor list
        if (child_count >= 2 || node == root) {
            ancestors.push_back(node);
        }

        for (int neighbor : adj[node]) {
            if (neighbor != parent && !visited[neighbor]) {
                dfs(neighbor, node, ancestors);
            }
        }
    };

    dfs(root, -1, {});  // start DFS from root with empty ancestor list
    return ancestor_list;
}
