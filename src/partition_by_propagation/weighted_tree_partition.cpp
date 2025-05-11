#include "weighted_tree_partition.h"

std::unordered_map<int, int> weighted_tree_partition(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E_mst,
    const std::unordered_map<int, int>& labeled_points
) {
    // Step 1: 构建邻接表，附带权重
    std::unordered_map<int, std::vector<std::pair<int, double>>> graph;
    for (int i = 0; i < E_mst.rows(); ++i) {
        int u = E_mst(i, 0);
        int v = E_mst(i, 1);
        double weight = (V.row(u) - V.row(v)).norm();

        graph[u].emplace_back(v, weight);
        graph[v].emplace_back(u, weight); // 因为是无向树
    }

    // Step 2: 多源 Dijkstra 初始化
    std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<QueueElement>> pq;
    std::unordered_map<int, int> node_to_class;
    std::unordered_set<int> visited;

    for (const auto& [node, class_id] : labeled_points) {
        pq.push({0.0, node, class_id});
    }

    // Step 3: 执行 Dijkstra
    while (!pq.empty()) {
        QueueElement current = pq.top();
        pq.pop();

        if (visited.count(current.node)) continue;
        visited.insert(current.node);
        node_to_class[current.node] = current.class_id;

        for (const auto& [neighbor, weight] : graph[current.node]) {
            if (!visited.count(neighbor)) {
                pq.push({current.dist + weight, neighbor, current.class_id});
            }
        }
    }

    return node_to_class;
}
