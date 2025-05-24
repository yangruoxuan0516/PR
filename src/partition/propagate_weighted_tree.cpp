#include "propagate_weighted_tree.h"


struct QueueElement {
    double dist;
    int node;
    int class_id;
    int parent;  // 👈 用于追踪来源点

    bool operator>(const QueueElement& other) const {
        return dist > other.dist;
    }
};

std::unordered_map<int, int> pointwise_partition_with_dijkstra(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E_mst,
    const std::unordered_map<int, int>& labeled_points
) {
    std::cout<<"pointwise_partition_with_dijkstra" << std::endl;
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
        QueueElement qe;
        qe.dist = 1.0;
        qe.node = node;
        qe.class_id = class_id;
        qe.parent = -1;
        pq.push(qe);
    }

    // Step 3: 执行 Dijkstra
    while (!pq.empty()) {
        QueueElement current = pq.top();
        pq.pop();

        if (visited.count(current.node)) continue; // visited.count(x) 是 std::unordered_set 的一个方法，用来检查集合中是否包含元素 x。
        visited.insert(current.node);
        node_to_class[current.node] = current.class_id;

        for (const auto& [neighbor, _] : graph[current.node]) {
            if (!visited.count(neighbor)) {
                double base_weight = (V.row(current.node) - V.row(neighbor)).norm();
                double angle_penalty = 1.0;
        
                if (current.parent != -1) {
                    // compute vectors
                    Eigen::Vector3d u = V.row(current.node) - V.row(current.parent);
                    Eigen::Vector3d v = V.row(neighbor) - V.row(current.node);
        
                    double norm_u = u.norm();
                    double norm_v = v.norm();
                
                    if (norm_u > 1e-8 && norm_v > 1e-8) {
                        double cos_angle = u.dot(v) / (norm_u * norm_v);
                        // Clamp cos_angle to [-1, 1] to avoid acos domain error
                        cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
                        double angle = std::acos(cos_angle);
                        angle_penalty = 1.0 + std::pow(angle / M_PI, 2);
                    } else {
                        angle_penalty = 1.0;  // no penalty if angle can't be computed
                    }
                }

                // print graph[current.node].size()
                // std::cout << "Node: " << current.node << ", Neighbors: " << graph[current.node].size() << std::endl;

                bool is_branching = graph[current.node].size() > 2;
                double branching_penalty = is_branching ? 10.0 : 1.0;  // 惩罚单链路切割
        
                double weight = base_weight * std::pow(angle_penalty, 10) * branching_penalty;

                // print current.dist
                // std::cout << "Current distance: " << current.dist << std::endl;

                // // print weight
                // std::cout << "Weight: " << weight << std::endl;

                // // print base_weight
                // std::cout << "--- Base weight: " << base_weight << std::endl;

                // // print angle_penalty
                // std::cout << "--- Angle penalty: " << std::pow(angle_penalty, 10) << std::endl;

                // // print branching_penalty
                // std::cout << "--- Branching penalty: " << branching_penalty << std::endl;
        
                pq.push({current.dist * weight, neighbor, current.class_id, current.node});
            }
        }
        
    }

    return node_to_class;
}





struct Segment {
    std::vector<int> indices;      // 点索引组成的路径
    Eigen::Vector3d vec;           // 首尾方向向量
    double length;                 // 该段总长度
    std::vector<int> neighbors;    // 邻接 segment 的索引
    int label = -1;                // 被传播的标签
    double cost = std::numeric_limits<double>::max(); // Dijkstra 花费
};

struct pair_hash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ std::hash<int>()(p.second << 1);
    }
};

double compute_segment_length(const std::vector<int>& indices, const Eigen::MatrixXd& V) {
    double len = 0.0;
    for (size_t i = 1; i < indices.size(); ++i) {
        len += (V.row(indices[i]) - V.row(indices[i - 1])).norm();
    }
    return len;
}

double angle_penalty(const Eigen::Vector3d& a, const Eigen::Vector3d& b, double lambda = 1.0) {
    double dot = a.normalized().dot(b.normalized());
    dot = std::max(-1.0, std::min(1.0, dot)); // Clamp
    return lambda * (1.0 - dot);  // dot 越小表示角度越大
}

std::unordered_map<int, int> segment_based_partition_based_on_dijkstra(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E_mst,
    const std::unordered_map<int, int>& labeled_points,
    Eigen::MatrixXd* debug_colors
) {
    std::cout<<"segment_based_partition_based_on_dijkstra" << std::endl;
    // Step 1: 构建无向邻接表
    std::unordered_map<int, std::vector<int>> graph;
    for (int i = 0; i < E_mst.rows(); ++i) {
        int u = E_mst(i, 0);
        int v = E_mst(i, 1);
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    // Step 2: 构建所有 segments
    std::vector<Segment> segments;
    std::unordered_map<int, std::vector<int>> node_to_segments; // node_id -> segment indices
    std::unordered_set<std::pair<int, int>, pair_hash> visited_edges;

    // std::vector<Segment> segments;
    // std::unordered_set<int> visited;
    for (const auto& [node, neighbors] : graph) {
        if (graph[node].size() == 2) continue; // 只从分叉点/叶子开始
    
        for (int neighbor : neighbors) {
            auto edge = std::minmax(node, neighbor);
            if (visited_edges.count(edge)) continue;
    
            // 新建 segment，从 node 开始沿着 neighbor 走
            Segment seg;
            seg.indices.push_back(node);
            int current = neighbor, prev = node;
    
            while (true) {
                seg.indices.push_back(current);
                visited_edges.insert(std::minmax(prev, current));
    
                if (graph[current].size() != 2) break;
    
                for (int next : graph[current]) {
                    if (next != prev) {
                        prev = current;
                        current = next;
                        break;
                    }
                }
            }
    
            // 保存 segment
            seg.vec = V.row(seg.indices.back()) - V.row(seg.indices.front());
            seg.length = compute_segment_length(seg.indices, V);
    
            int seg_id = segments.size();
            for (int idx : seg.indices)
                node_to_segments[idx].push_back(seg_id);
    
            segments.push_back(seg);
        }
    }
    


    if (debug_colors) {
        debug_colors->resize(V.rows(), 3);
        debug_colors->setZero();
        for (int i = 0; i < segments.size(); ++i) {
            Eigen::RowVector3d color = Eigen::RowVector3d::Random().cwiseAbs();
            color /= color.maxCoeff(); // 归一化为 [0, 1]
            for (int idx : segments[i].indices) {
                debug_colors->row(idx) = color;
            }
        }
    }


    // Step 3: 构建 segment graph（根据共有点连边）
    for (int i = 0; i < segments.size(); ++i) {
        std::unordered_set<int> neighbor_ids;
        for (int node : segments[i].indices) {
            for (int seg_j : node_to_segments[node]) {
                if (seg_j != i) neighbor_ids.insert(seg_j);
            }
        }
        segments[i].neighbors.assign(neighbor_ids.begin(), neighbor_ids.end());
    }

    // Step 4: 初始化 Dijkstra 的队列（从 labeled_points 所在 segment 出发）
    using State = std::pair<double, int>; // (cost, segment_id)
    std::priority_queue<State, std::vector<State>, std::greater<>> pq;

    for (const auto& [idx, label] : labeled_points) {
        for (int seg_id : node_to_segments[idx]) {
            if (segments[seg_id].cost > 0) {
                segments[seg_id].cost = 0;
                segments[seg_id].label = label;
                pq.emplace(0, seg_id);
            }
        }
    }

    // Step 5: Dijkstra 传播标签
    while (!pq.empty()) {
        auto [cost_u, u] = pq.top(); pq.pop();
        if (cost_u > segments[u].cost) continue;

        for (int v : segments[u].neighbors) {
            double len = segments[u].length + segments[v].length;
            double angle = angle_penalty(segments[u].vec, segments[v].vec);
            double new_cost = segments[u].cost + len + angle;

            if (new_cost < segments[v].cost) {
                segments[v].cost = new_cost;
                segments[v].label = segments[u].label;
                pq.emplace(new_cost, v);
            }
        }
    }

    // Step 6: 将 segment 中的点赋 label
    std::unordered_map<int, int> node_to_class;
    for (const auto& seg : segments) {
        for (int idx : seg.indices) {
            node_to_class[idx] = seg.label;
        }
    }

    // Step 7: 确保 labeled_points 自己也写入
    for (const auto& [idx, label] : labeled_points) {
        node_to_class[idx] = label;
    }

    // Step 8: 报告遗漏
    int missing = 0;
    for (int i = 0; i < V.rows(); ++i) {
        if (!node_to_class.count(i)) ++missing;
    }

    return node_to_class;
}
