#include "connect_points/mst.h"

// 最小生成树（Kruskal）：从 Delaunay 边集提取 MST
Eigen::MatrixXi extract_mst_from_delaunay(
    const Eigen::MatrixXd& V,            // 点集
    const Eigen::MatrixXi& E_delaunay    // Delaunay 边集（索引）
) {
    struct Edge {
        int u, v;
        double weight;
        bool operator<(const Edge& other) const { return weight < other.weight; }
    };

    // Step 1: 构建边集（带权重）
    std::vector<Edge> edges;
    for (int i = 0; i < E_delaunay.rows(); ++i) {
        int u = E_delaunay(i, 0);
        int v = E_delaunay(i, 1);
        double w = (V.row(u) - V.row(v)).norm();
        edges.push_back({u, v, w});
    }

    std::sort(edges.begin(), edges.end());

    // Step 2: 初始化并查集
    int n = V.rows();
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);

    auto find = [&](int x) {
        while (x != parent[x]) x = parent[x] = parent[parent[x]];
        return x;
    };

    auto unite = [&](int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        parent[px] = py;
        return true;
    };

    // Step 3: Kruskal 主循环
    std::vector<Eigen::Vector2i> mst_edges;
    for (const auto& e : edges) {
        if (unite(e.u, e.v)) {
            mst_edges.emplace_back(e.u, e.v);
            if (mst_edges.size() == n - 1) break; // 最多 n-1 条边
        }
    }

    // 转为 Eigen::MatrixXi
    Eigen::MatrixXi E_mst(mst_edges.size(), 2);
    for (int i = 0; i < mst_edges.size(); ++i) {
        E_mst.row(i) = mst_edges[i];
    }

    return E_mst;
}
