#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject.h>
#include <fstream>
#include <iostream>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <Eigen/Dense>
#include <vector>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <cstdlib>
#include <ctime>

#include "connect_points/travel_salesman.h"
#include "connect_points/snake.h"
#include "connect_points/delaunay.h"
#include "connect_points/dijkstra.h"
#include "connect_points/mst.h"

#include "tree_hierarchy/find_hierarchy_with_root.h"

#include "params.h"

Eigen::RowVector3d default_color(0.5, 0.5, 0.5); 

Eigen::RowVector3d hsv2rgb(double h, double s, double v) {
    double c = v * s;
    double x = c * (1 - std::abs(fmod(h / 60.0, 2) - 1));
    double m = v - c;
    double r, g, b;

    if (h < 60)       r = c, g = x, b = 0;
    else if (h < 120) r = x, g = c, b = 0;
    else if (h < 180) r = 0, g = c, b = x;
    else if (h < 240) r = 0, g = x, b = c;
    else if (h < 300) r = x, g = 0, b = c;
    else              r = c, g = 0, b = x;

    return Eigen::RowVector3d(r + m, g + m, b + m);
}

Eigen::RowVector3d generate_distinct_color(int index, int total = 20) {
    double h = fmod((index * 360.0 / total), 360.0);  // 均匀分布在色相环上
    double s = 0.8;  // 高饱和
    double v = 0.9;  // 高亮度
    return hsv2rgb(h, s, v);
}



bool loadXYZ(const std::string& filename, Eigen::MatrixXd& V) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }

    std::vector<Eigen::Vector3d> points;
    double x, y, z;
    while (infile >> x >> y >> z) {
        points.push_back(Eigen::Vector3d(x, y, z));
    }
    infile.close();

    // Convert to Eigen matrix
    V.resize(points.size(), 3);
    for (size_t i = 0; i < points.size(); ++i) {
        V.row(i) = points[i];
    }
    return true;
}

bool click_point(igl::opengl::glfw::Viewer& viewer,
    Eigen::MatrixXd& V,
    Eigen::MatrixXd& C,
    int button,
    int modifier,
    const Eigen::RowVector3d& selected_color)
{
    int vid = -1;
    double min_dis = 20;  // pixel threshold
    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y - 1;

    Eigen::Vector3f click_pos(x, y, 0);

    for (int i = 0; i < V.rows(); i++) {
        Eigen::Vector3f projected;
        igl::project(V.row(i).cast<float>(), viewer.core().view, viewer.core().proj, viewer.core().viewport, projected);

        double dist = (projected.head<2>() - click_pos.head<2>()).norm();
        if (dist < min_dis) {
            min_dis = dist;
            vid = i;
        }
    }

    if (vid != -1) {
        if (C.row(vid) == default_color) {
            C.row(vid) = selected_color;
        } 
        else {
            C.row(vid) = default_color;
        }
        viewer.data_list[0].set_points(V, C);
        return true;
    }

    return false;
}



Eigen::MatrixXd get_colored_points(const Eigen::MatrixXd& V, const Eigen::MatrixXd& C, const Eigen::RowVector3d& color) {
    std::vector<int> selected_points;

    for (int i = 0; i < V.rows(); i++) {
        if (C.row(i) == color) {
            selected_points.push_back(i);
        }
    }

    if (selected_points.size() < 2) {
        return Eigen::MatrixXd();
    }

    Eigen::MatrixXd points(selected_points.size(), 3);

    for (int i = 0; i < selected_points.size(); i++) {
        points.row(i) = V.row(selected_points[i]);
    }
    return points;
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> connect_points_with_snake(GUIParams& params, const Eigen::MatrixXd& V, Eigen::MatrixXd points, const Eigen::RowVector3d& color, igl::opengl::glfw::Viewer& viewer) {
    Eigen::MatrixXd V_raw;
    Eigen::MatrixXi E_raw;

    std::tie(V_raw, E_raw) = travel_salesman(points); // V is the selected points
                                                      // E is the connecting order

    Eigen::MatrixXd V_final;
    Eigen::MatrixXi E_final;                                  

    // for each edge (every two suscessive points), call snake
    for (int i = 0; i < E_raw.rows(); i++) {
        Eigen::MatrixXd V_;
        Eigen::MatrixXi E_;
        std::tie(V_, E_) = snake(params, V, V_raw.row(E_raw(i, 0)), V_raw.row(E_raw(i, 1)), viewer, color);
        if (i == 0) {
            V_final = V_;
            E_final = E_;
        } else {
            int vertex_offset = V_final.rows();  

            // Append vertices
            V_final.conservativeResize(vertex_offset + V_.rows(), V_final.cols());
            V_final.bottomRows(V_.rows()) = V_;

            // Append edges, with correct offset
            E_final.conservativeResize(E_final.rows() + E_.rows(), E_final.cols());
            E_final.bottomRows(E_.rows()) = E_.array() + vertex_offset;
        }
    }

    return std::make_tuple(V_final,E_final);

}


int main() {
// --- load the point cloud
    Eigen::MatrixXd V;
    // std::string filename = "/Users/ruox/Documents/DoubleDegree/cours_2/ParcoursRecherche/projet/generate_example_point_cloud/point_cloud/X_form_C.xyz";
    std::string filename = "/Users/ruox/Documents/DoubleDegree/cours_2/ParcoursRecherche/projet/python/skeleton.xyz";

    if (!loadXYZ(filename, V)) return 1;
    Eigen::MatrixXd C(V.rows(), 3);
    for (int i = 0; i < V.rows(); ++i) {
        C.row(i) = default_color;
    }
    Eigen::MatrixXi E;

// --- colors
    std::vector<std::string> type_labels = {"Type 1"};
    int current_type_index = 0;  // Initially selected type
    std::vector<Eigen::RowVector3d> type_colors = {
        generate_distinct_color(current_type_index - 1)
    };

// -- delaunay triangulation
    // raw
    Delaunay dt;
    insert_points_into_delaunay(V, dt);

    Eigen::MatrixXi E_dt;
    extract_edges_from_delaunay(dt, V, E_dt);
    Eigen::MatrixXi E_dt_filtered = E_dt; 
    // edge length
    std::vector<double> dt_edge_lengths;
    for (int i = 0; i < E_dt.rows(); ++i)
    {
        int idx1 = E_dt(i,0);
        int idx2 = E_dt(i,1);
        double length = (V.row(idx1) - V.row(idx2)).norm();
        dt_edge_lengths.push_back(length);
    }
    float max_edge_length = *std::max_element(dt_edge_lengths.begin(), dt_edge_lengths.end());
    float filter_edge_length = max_edge_length;

// --- mst
    Eigen::MatrixXi E_mst = extract_mst_from_delaunay(V, E_dt);

// --- find root
    int root = find_root(V, E_mst);
    // print the root point
    std::cout << "Root point: " << root << std::endl;

// --- viewer
    igl::opengl::glfw::Viewer viewer;
    viewer.core().background_color = Eigen::Vector4f(1.0, 1.0, 1.0, 1.0);  // R, G, B, A

    viewer.append_mesh(); // data_id = 0 for static original points
    viewer.append_mesh(); // data_id = 1 for dynamic KNN points
    viewer.append_mesh(); // data_id = 2 for dynamic curve points
    viewer.append_mesh(); // data_id = 3 for delaunay edges
    viewer.append_mesh(); // data_id = 4 for delaunay edges between selected points
    viewer.append_mesh(); // data_id = 5 for mst
    viewer.append_mesh(); // data_id = 6 for mst edges between selected points
    viewer.append_mesh(); // data_id = 7 for root point
    
    viewer.data_list[0].point_size = 5; 
    viewer.data_list[0].set_points(V, C);

    viewer.core().align_camera_center(V);
    viewer.core().camera_eye = Eigen::Vector3f(0, 5, 0); // Set camera position
    viewer.core().camera_up = Eigen::Vector3f(0, 0, 1); 

    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int button, int modifier) {
        return click_point(viewer, V, C, button, modifier, type_colors[current_type_index]);
    };


// --- menu
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

    GUIParams params; 
    
    bool show_delaunay = false;
    bool show_delaunay_selected = false;
    bool show_mst = false;
    bool show_mst_selected = false;
    bool show_root = false;

    menu.callback_draw_viewer_menu = [&]()
    {
        ImGui::SetNextWindowSize(ImVec2(350, 750), ImGuiCond_FirstUseEver); // width, height
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::Begin("Menu", nullptr, ImGuiWindowFlags_NoCollapse);

        ImGui::Text("Demostration Settings:");
        ImGui::SliderFloat("Point radius", &viewer.data_list[0].point_size, 0.0f, 10.0f);

        ImGui::Separator();

        ImGui::Text("Snake Params:");
        ImGui::SliderInt("snake iteration num", &params.snake_iteration_num, 0, 100);
        ImGui::SliderFloat("snake step", &params.snake_step, 0.0f, 0.2f);
        ImGui::SliderInt("snake resample num", &params.snake_resample_num, 0, 100);
        ImGui::SliderFloat("weight elastic", &params.weight_elastic, 0.0f, 10.0f);
        ImGui::SliderFloat("weight curvature", &params.weight_curvature, 0.0f, 10.0f);
        ImGui::SliderFloat("weight attraction", &params.weight_attraction, 0.0f, 100.0f);
        

        ImGui::Separator();
        ImGui::Text("Point Type Selection:");
    
        // Render each type as a selectable button
        for (int i = 0; i < type_labels.size(); ++i) {
            if (ImGui::Selectable(type_labels[i].c_str(), current_type_index == i)) {
                current_type_index = i;
            }
        }
    
        if (ImGui::Button("Add new type")) {
            int new_index = type_labels.size() + 1;
            type_labels.push_back("Type " + std::to_string(new_index));
            type_colors.push_back(generate_distinct_color(new_index - 1));  // New random color
            current_type_index = new_index - 1;  // Select the newly added type
        }

        if (ImGui::Button("Reset", ImVec2(-1, 0))) {
            viewer.data_list[0].clear(); 
            viewer.data_list[1].clear(); 
            viewer.data_list[2].clear();
            viewer.data_list[3].clear();
            viewer.data_list[4].clear();
            viewer.data_list[5].clear();
            viewer.data_list[6].clear();
            for (int i = 0; i < V.rows(); ++i) {
                C.row(i) = default_color;
            }
            viewer.data_list[0].set_points(V, C);
            type_labels = {"Type 1"};
            current_type_index = 0;
            type_colors = { generate_distinct_color(current_type_index - 1) };
        }

        ImGui::Separator();
        ImGui::Text("Snake Connection:");

        if (ImGui::Button("connect with snake", ImVec2(-1, 0))) { 
            std::vector<Eigen::RowVector3d> all_colors;
            Eigen::MatrixXi E_all(0, 2);
            Eigen::MatrixXd V_all(0, 3);


            for (int i = 0; i < type_labels.size(); i++) {
                Eigen::MatrixXd V_temp = V;
                Eigen::MatrixXd colored_points = get_colored_points(V_temp, C, type_colors[i]);
                if (colored_points.rows() < 2) {
                    continue;
                }
                auto [V_result, E_result] = connect_points_with_snake(params, V_temp, colored_points, type_colors[i], viewer);

                int old_V_rows = V_all.rows();
                int old_E_rows = E_all.rows();
                if (V_result.rows() > 0) {
                    // Concatenate V to V_all
                    V_all.conservativeResize(old_V_rows + V_result.rows(), 3);
                    V_all.bottomRows(V_result.rows()) = V_result;
                }

                if (E_result.rows() > 0) {
                    // Concatenate E to E_all
                    E_all.conservativeResize(old_E_rows + E_result.rows(), 2);
                    E_all.bottomRows(E_result.rows()) = E_result.array() + old_V_rows;

                    // Save color for each edge
                    for (int j = 0; j < E_result.rows(); ++j) {
                        all_colors.push_back(type_colors[i]);
                    }
                }
            }

            if (E_all.rows() > 0) {
                Eigen::MatrixXd C_all(E_all.rows(), 3);
                for (int i = 0; i < all_colors.size(); ++i) {
                    C_all.row(i) = all_colors[i];
                }
                viewer.data_list[2].set_edges(V_all, E_all, C_all); 
            }
        }

        if (ImGui::Button("One Step Optimize", ImVec2(-1, 0))) {
            optimize_snake_step();
        }
        if (ImGui::Button("One Iteration Optimize", ImVec2(-1, 0))) {
            optimize_snake_iteration();
        }
        if (ImGui::Button("Complete Optimize", ImVec2(-1, 0))) {
            optimize_snake_complete();
        }
        if (ImGui::Button("Unshow snake", ImVec2(-1, 0))) {
            viewer.data_list[2].clear(); 
        }



        ImGui::Separator();
        ImGui::Text("Delaunay Triangulation:");

        if (ImGui::Button("Delaunay Triangulation", ImVec2(-1, 0))) {
            show_delaunay = !show_delaunay;
            if (show_delaunay) {
                viewer.data_list[3].set_edges(V, E_dt_filtered, Eigen::RowVector3d(0.0, 0.0, 0.0));
            }
            else {
                viewer.data_list[3].clear(); 
            }
        }

        // get max in dt_edge_lengths
        bool updated = ImGui::SliderFloat("max edge length", &filter_edge_length, 0.0f, max_edge_length);
        
        if (updated) {
            std::vector<Eigen::Vector2i> filtered_edges;
            for (int i = 0; i < E_dt.rows(); ++i)
            {
                double length = dt_edge_lengths[i];
                if (length <= filter_edge_length) 
                {
                    filtered_edges.push_back(E_dt.row(i));
                }
            }
            E_dt_filtered.resize(filtered_edges.size(), 2);
            for (int i = 0; i < filtered_edges.size(); ++i)
            {
                E_dt_filtered.row(i) = filtered_edges[i];
            }
            if (show_delaunay) {
                viewer.data_list[3].clear();
                viewer.data_list[3].set_edges(V, E_dt_filtered, Eigen::RowVector3d(0.0, 0.0, 0.0)); 
            }
            else
            {
                viewer.data_list[3].clear(); // clear edges to hide
            }
        }

        if (ImGui::Button("connect with dijkstra (delaunay)", ImVec2(-1, 0))) {
            std::vector<Eigen::RowVector3d> all_colors;
            Eigen::MatrixXi E_all(0, 2);
        
            auto graph = build_graph_from_edges(V, E_dt_filtered);
        
            for (int i = 0; i < type_labels.size(); i++) {
                Eigen::MatrixXd V_temp = V;
                Eigen::MatrixXd colored_points = get_colored_points(V_temp, C, type_colors[i]);
        
                if (colored_points.rows() < 2) continue;
        
                for (int j = 0; j < colored_points.rows(); j++) {
                    for (int k = j + 1; k < colored_points.rows(); k++) {
                        int start = find_closest_point(V, colored_points.row(j));
                        int end = find_closest_point(V, colored_points.row(k));
                        if (start != -1 && end != -1) {

                            Eigen::MatrixXi E_dijkstra = dijkstra_edges(V.rows(), graph, start, end);


                            if (E_dijkstra.rows() > 0) {
                                // 累加边
                                int old_rows = E_all.rows();
                                E_all.conservativeResize(old_rows + E_dijkstra.rows(), 2);
                                E_all.bottomRows(E_dijkstra.rows()) = E_dijkstra;
                    
                                // 为每条边添加对应颜色
                                for (int p = 0; p < E_dijkstra.rows(); ++p) {
                                    all_colors.push_back(type_colors[i]);
                                }
                            }
                        }
                    }
                }
            }
            show_delaunay_selected = !show_delaunay_selected;
            if (show_delaunay_selected) {
                // 最后统一渲染一次
                if (E_all.rows() > 0) {
                    Eigen::MatrixXd C_all(E_all.rows(), 3);
                    for (int i = 0; i < all_colors.size(); ++i) {
                        C_all.row(i) = all_colors[i];
                    }
                    viewer.data_list[4].set_edges(V, E_all, C_all); 
                }
            }
            else {
                viewer.data_list[4].clear(); // clear edges to hide
            }
        }


        ImGui::Separator();
        ImGui::Text("Minimum Spanning Tree:");

        if (ImGui::Button("Minimum Spanning Tree", ImVec2(-1, 0))) {
            show_mst = !show_mst;
            if (show_mst) {
                viewer.data_list[5].set_edges(V, E_mst, Eigen::RowVector3d(0, 0, 0));
            }
            else {
                viewer.data_list[5].clear(); // clear edges to hide
            }
        }


        if (ImGui::Button("connect with dijkstra (mst)", ImVec2(-1, 0))) {
            std::vector<Eigen::RowVector3d> all_colors;
            Eigen::MatrixXi E_all(0, 2);
        
            auto graph = build_graph_from_edges(V, E_mst);
        
            for (int i = 0; i < type_labels.size(); i++) {
                Eigen::MatrixXd V_temp = V;
                Eigen::MatrixXd colored_points = get_colored_points(V_temp, C, type_colors[i]);
        
                if (colored_points.rows() < 2) continue;
        
                for (int j = 0; j < colored_points.rows(); j++) {
                    for (int k = j + 1; k < colored_points.rows(); k++) {
                        int start = find_closest_point(V, colored_points.row(j));
                        int end = find_closest_point(V, colored_points.row(k));
                        if (start != -1 && end != -1) {

                            Eigen::MatrixXi E_dijkstra = dijkstra_edges(V.rows(), graph, start, end);


                            if (E_dijkstra.rows() > 0) {
                                // 累加边
                                int old_rows = E_all.rows();
                                E_all.conservativeResize(old_rows + E_dijkstra.rows(), 2);
                                E_all.bottomRows(E_dijkstra.rows()) = E_dijkstra;
                    
                                // 为每条边添加对应颜色
                                for (int p = 0; p < E_dijkstra.rows(); ++p) {
                                    all_colors.push_back(type_colors[i]);
                                }
                            }
                        }
                    }
                }
            }
            show_mst_selected = !show_mst_selected;
            if (show_mst_selected) {
                // 最后统一渲染一次
                if (E_all.rows() > 0) {
                    Eigen::MatrixXd C_all(E_all.rows(), 3);
                    for (int i = 0; i < all_colors.size(); ++i) {
                        C_all.row(i) = all_colors[i];
                    }
                    viewer.data_list[6].set_edges(V, E_all, C_all); 
                }
            }
            else {
                viewer.data_list[6].clear(); // clear edges to hide
            }
        }

        ImGui::Separator();
        ImGui::Text("Tree Hierarchy:");
        if (ImGui::Button("Find Root", ImVec2(-1, 0))) {
            if (show_root) {
                show_root = false;
                viewer.data_list[7].clear(); // clear edges to hide
            } else {
                show_root = true;
                viewer.data_list[7].set_points(V.row(root), Eigen::RowVector3d(1, 0, 0)); 
            }   
        }


        ImGui::End(); 
    };

    viewer.launch();

    return 0;
}
