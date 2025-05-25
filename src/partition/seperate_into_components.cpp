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



std::vector<ComponentGraph> get_component_graphs(
    const Eigen::MatrixXd& V, double radius
) {
    std::vector<ComponentGraph> component_graphs;

    // ---- Step 1: 调用你的原始连通 component 提取 ----
    std::vector<std::vector<int>> components = seperate_into_components(V, radius);

    // ---- Step 2: 构建每个 component 的图结构 ----
    for (int cid = 0; cid < components.size(); ++cid) {
        const auto& comp = components[cid];

        Eigen::MatrixXd V_sub(comp.size(), 3);
        Eigen::VectorXi global_indices(comp.size());

        for (int i = 0; i < comp.size(); ++i) {
            V_sub.row(i) = V.row(comp[i]);
            global_indices[i] = comp[i];
        }

        // Delaunay triangulation (XY平面，可自定义)
        Delaunay dt;
        insert_points_into_delaunay(V_sub, dt);

        Eigen::MatrixXi E_dt;
        extract_edges_from_delaunay(dt, V_sub, E_dt);

        Eigen::MatrixXi E_mst = extract_mst_from_delaunay(V_sub, E_dt);

        Eigen::RowVector3d color = generate_distinct_color(cid, components.size());

        component_graphs.push_back({
            V_sub,
            global_indices,
            E_mst,
            color,
            cid  // 可选的 component ID
        });
    }

    return component_graphs;
}


void component_graph_vertices(
    const std::vector<ComponentGraph>& component_graphs,
    Eigen::MatrixXd& V_out,
    Eigen::MatrixXd& C_out
) {
    int total = 0;
    for (const auto& cg : component_graphs)
        total += cg.global_indices.size();

    V_out.resize(total, 3);
    C_out.resize(total, 3);

    int cursor = 0;
    for (const auto& cg : component_graphs) {
        for (int i = 0; i < cg.global_indices.size(); ++i) {
            V_out.row(cursor) = cg.V_sub.row(i);
            C_out.row(cursor) = cg.color;
            cursor++;
        }
    }
}


void component_graph_edges(
    const std::vector<ComponentGraph>& component_graphs,
    Eigen::MatrixXd& P1_out,
    Eigen::MatrixXd& P2_out,
    Eigen::MatrixXd& C_out
) {
    int total_edges = 0;
    for (const auto& cg : component_graphs)
        total_edges += cg.E_mst.rows();

    P1_out.resize(total_edges, 3);
    P2_out.resize(total_edges, 3);
    C_out.resize(total_edges, 3);

    int cursor = 0;
    for (const auto& cg : component_graphs) {
        for (int i = 0; i < cg.E_mst.rows(); ++i) {
            int u = cg.E_mst(i, 0);
            int v = cg.E_mst(i, 1);

            P1_out.row(cursor) = cg.V_sub.row(u);
            P2_out.row(cursor) = cg.V_sub.row(v);
            C_out.row(cursor)  = cg.color;
            cursor++;
        }
    }
}
