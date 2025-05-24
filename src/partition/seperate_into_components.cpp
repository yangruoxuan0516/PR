// seperate_into_components.cpp
#include "seperate_into_components.h"

std::vector<std::vector<int>> seperate_into_components(const Eigen::MatrixXd& V, double radius) {
    int N = V.rows();
    std::vector<std::vector<int>> adjacency(N);
    std::vector<bool> visited(N, false);
    std::vector<std::vector<int>> components;

    // Brute-force radius neighbors
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if ((V.row(i) - V.row(j)).norm() <= radius) {
                adjacency[i].push_back(j);
                adjacency[j].push_back(i);
            }
        }
    }

    // BFS component labeling
    for (int i = 0; i < N; ++i) {
        if (visited[i]) continue;

        std::vector<int> comp;
        std::queue<int> q;
        q.push(i);
        visited[i] = true;

        while (!q.empty()) {
            int cur = q.front(); q.pop();
            comp.push_back(cur);
            for (int nei : adjacency[cur]) {
                if (!visited[nei]) {
                    visited[nei] = true;
                    q.push(nei);
                }
            }
        }

        components.push_back(comp);
    }

    return components;
}
