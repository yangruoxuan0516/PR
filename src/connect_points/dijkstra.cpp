#include "connect_points/dijkstra.h"

const double INF = std::numeric_limits<double>::infinity();

// 构建邻接表：从点集和边集构建图
std::vector<std::vector<std::pair<int, double>>> build_graph_from_edges(
    const Eigen::MatrixXd& V_dt,
    const Eigen::MatrixXi& E_dt
) {
    int n = V_dt.rows();
    std::vector<std::vector<std::pair<int, double>>> graph(n);

    for (int i = 0; i < E_dt.rows(); ++i) {
        int u = E_dt(i, 0);
        int v = E_dt(i, 1);
        double weight = (V_dt.row(u) - V_dt.row(v)).norm(); // 欧几里得距离
        // filter with edge length
        if (weight > 100) continue; // 0.1 is the threshold
        graph[u].push_back({v, weight});
        graph[v].push_back({u, weight}); // 无向图
    }
    return graph;
}


int find_closest_point(const Eigen::MatrixXd& V, const Eigen::RowVectorXd& query_point) {
    int index = -1;
    double min_dist = std::numeric_limits<double>::infinity();
    for (int i = 0; i < V.rows(); ++i) {
        double dist = (V.row(i) - query_point).norm();
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }
    return index;
}


Eigen::MatrixXi dijkstra_edges(
    int n,
    const std::vector<std::vector<std::pair<int, double>>>& graph,
    int start,
    int end
) {
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dist(n, INF);
    std::vector<int> prev(n, -1);
    dist[start] = 0.0;

    using P = std::pair<double, int>; // (distance, node)
    std::priority_queue<P, std::vector<P>, std::greater<>> pq;
    pq.push({0.0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (u == end) break;
        if (d > dist[u]) continue;

        for (auto [v, weight] : graph[u]) {
            if (dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                prev[v] = u;
                pq.push({dist[v], v});
            }
        }
    }

    // reconstruct edge path
    std::vector<std::pair<int, int>> edges;
    for (int at = end; prev[at] != -1; at = prev[at]) {
        edges.emplace_back(prev[at], at);
    }
    std::reverse(edges.begin(), edges.end());

    Eigen::MatrixXi E(edges.size(), 2);
    for (int i = 0; i < edges.size(); ++i) {
        E(i, 0) = edges[i].first;
        E(i, 1) = edges[i].second;
    }
    return E;
}
