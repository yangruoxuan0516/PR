#include <iostream>
#include "connect_points/travel_salesman.h"

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> travel_salesman(const Eigen::MatrixXd& points) {

    int n = points.rows();

    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0));

    // Compute pairwise distances
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                dist[i][j] = (points.row(i) - points.row(j)).norm();
            }
        }
    }

    std::vector<Eigen::RowVector2i> edges;
    // DP table: dp[mask][i] = min cost to visit all nodes in mask ending at i
    std::vector<std::vector<double>> dp(1 << n, std::vector<double>(n, std::numeric_limits<double>::infinity()));
    std::vector<std::vector<int>> parent(1 << n, std::vector<int>(n, -1));
    // Initialize base cases
    for (int i = 0; i < n; i++) {
        dp[1 << i][i] = 0;  // Only visiting i
    }
    // Iterate over all subsets of nodes
    for (int mask = 1; mask < (1 << n); mask++) {
        for (int i = 0; i < n; i++) {
            if (!(mask & (1 << i))) continue;  // If i is not in the subset, skip

            for (int j = 0; j < n; j++) {
                if (mask & (1 << j)) continue;  // If j is already in the subset, skip

                int new_mask = mask | (1 << j);
                double new_cost = dp[mask][i] + dist[i][j];

                if (new_cost < dp[new_mask][j]) {
                    dp[new_mask][j] = new_cost;
                    parent[new_mask][j] = i;
                }
            }
        }
    }
    // Find the best ending point for the minimum path
    double min_cost = std::numeric_limits<double>::infinity();
    int last_index = -1, final_mask = (1 << n) - 1;

    for (int i = 0; i < n; i++) {
        if (dp[final_mask][i] < min_cost) {
            min_cost = dp[final_mask][i];
            last_index = i;
        }
    }
    // Backtrack to construct the path
    
    int mask = final_mask;
    while (last_index != -1) {
        int prev_index = parent[mask][last_index];
        if (prev_index != -1) {
            edges.push_back(Eigen::RowVector2i(prev_index, last_index));
        }
        mask ^= (1 << last_index);
        last_index = prev_index;
    }

    Eigen::MatrixXi E(edges.size(), 2);
    for (size_t i = 0; i < edges.size(); i++) {
        E.row(i) = edges[i];
    }

    return std::make_tuple(points,E);
}