#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject.h>
#include <igl/edges.h>
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
#include <limits> // for quiet_NaN()

#include "connect_points/travel_salesman.h"
#include "connect_points/snake.h"
#include "connect_points/delaunay.h"
#include "connect_points/dijkstra.h"
#include "connect_points/mst.h"

#include "tree_hierarchy/find_hierarchy_with_root.h"
#include "tree_hierarchy/seperate_based_on_hierarchy.h"

#include "partition/propagate_weighted_tree.h"
#include "partition/seperate_into_components.h"

#include "params.h"
#include "utils/color_utils.h"



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

void update_selected_layer(
    igl::opengl::glfw::Viewer& viewer,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& C,
    Eigen::RowVector3d default_color,
    int data_id)
{
    std::vector<int> selected_indices;
    for (int i = 0; i < C.rows(); ++i) {
        if (C.row(i) != default_color) {
            selected_indices.push_back(i);
        }
    }

    Eigen::MatrixXd V_sel(selected_indices.size(), 3);
    Eigen::MatrixXd C_sel(selected_indices.size(), 3);
    for (int i = 0; i < selected_indices.size(); ++i) {
        V_sel.row(i) = V.row(selected_indices[i]);
        C_sel.row(i) = C.row(selected_indices[i]);
    }

    viewer.data_list[data_id].clear();
    viewer.data_list[data_id].set_points(V_sel, C_sel);
    viewer.data_list[data_id].point_size = 10;
}


bool click_point(igl::opengl::glfw::Viewer& viewer,
    Eigen::MatrixXd& V,
    Eigen::MatrixXd& C,
    int button,
    int modifier,
    const Eigen::RowVector3d& selected_color,
    std::vector<Eigen::RowVector3d> type_colors,
    Eigen::RowVector3d default_color,
    int& index_selected_points,
    std::vector<int> point_to_component_id)
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
            // print the component id
            std::cout << "Component ID: " << point_to_component_id[vid] << std::endl;

// --- this part is just for testing ---
            // Delaunay dt;
            // insert_points_into_delaunay(V, dt);
            // Eigen::MatrixXi E_dt;
            // extract_edges_from_delaunay(dt, V, E_dt);
            // Eigen::MatrixXi E_mst = extract_mst_from_delaunay(V, E_dt);
            // // color also the ancestors
            // int root = find_root(V, E_mst);
            // std::vector<std::vector<int>> ancestor_list = find_ancestor_list(V, E_mst, root);
    // -- here is the test for the ancestor list
            // std::vector<int> ancestors = ancestor_list[vid];
            // // print the ancestors
            // std::cout << "Anscestors of point " << vid << ": ";
            // for (int i = 0; i < ancestors.size(); i++) {
            //     std::cout << ancestors[i] << " ";
            // }
            // std::cout << std::endl;
            // for (int i = 0; i < ancestors.size(); i++) {
            //     C.row(ancestors[i]) = selected_color;
            // }
    // -- here is the test for the subtree
            // std::vector<int> current_subtree = find_subtree(V, E_mst, ancestor_list, vid);
            // // print the subtree
            // std::cout << "Subtree of point " << vid << ": ";
            // for (int i = 0; i < current_subtree.size(); i++) {
            //     std::cout << current_subtree[i] << " ";
            // }
            // std::cout << std::endl;
            // // color the subtree
            // for (int i = 0; i < current_subtree.size(); i++) {
            //     C.row(current_subtree[i]) = selected_color;
            // }
    // -- here is to test going up
            // std::vector<int> new_subtree = going_up(V, E_mst, ancestor_list, current_subtree);
            // // print the new subtree
            // std::cout << "New subtree of point " << vid << ": ";
            // for (int i = 0; i < new_subtree.size(); i++) {
            //     std::cout << new_subtree[i] << " ";
            // }
            // std::cout << std::endl;
            // // color the new subtree
            // for (int i = 0; i < new_subtree.size(); i++) {
            //     C.row(new_subtree[i]) = selected_color;
            // }
    // -- here is for testing the splitting based on the hierarchy, but it is not working
            // Eigen::MatrixXd new_colors = seperate_based_on_hierarchy(V, E_mst, C, type_colors, ancestor_list);
            // // print the new color length
            // std::cout << "New color length: " << new_colors.rows() << std::endl;
            // // print point cloud length
            // std::cout << "Point cloud length: " << V.rows() << std::endl;
            // viewer.data_list[index_pointwise_partition].set_points(V, new_colors);
// --- end of the testing code ---

        } 
        else {
            C.row(vid) = default_color;
        }
        update_selected_layer(viewer, V, C, default_color, index_selected_points);
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

    // if (selected_points.size() < 2) {
    //     return Eigen::MatrixXd();
    // }

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
    Eigen::MatrixXd V, V_ori;
    // std::string filename = "/Users/ruox/Documents/DoubleDegree/cours_2/ParcoursRecherche/projet/generate_example_point_cloud/point_cloud/X_form_C.xyz";
    std::string filename_ori = "/Users/ruox/Documents/DoubleDegree/cours_2/ParcoursRecherche/projet/example-project/data/points.xyz";
    // std::string filename = "/Users/ruox/Documents/DoubleDegree/cours_2/ParcoursRecherche/projet/python/skeleton_raw_0.05.xyz";
    std::string filename = "/Users/ruox/Documents/DoubleDegree/cours_2/ParcoursRecherche/projet/example-project/data/skeleton.xyz";



    Eigen::MatrixXd V_surface;  // 顶点
    Eigen::MatrixXi F_surface;  // 面
    igl::readOBJ("/Users/ruox/Documents/DoubleDegree/cours_2/ParcoursRecherche/projet/example-project/data/vessel_surface.obj", V_surface, F_surface);


    if (!loadXYZ(filename_ori, V_ori)) return 1;
    if (!loadXYZ(filename, V)) return 1;

    Eigen::MatrixXd default_C(V.rows(), 3);

    Eigen::RowVector3d default_color = Eigen::RowVector3d(1, 0, 0);

    Eigen::MatrixXd C(V.rows(), 3);
    for (int i = 0; i < V.rows(); ++i) {
        default_C.row(i) = default_color;
        C.row(i) = default_color;
    }

    Eigen::MatrixXd C_ori(V_ori.rows(), 3);
    for (int i = 0; i < V_ori.rows(); ++i) {
        C_ori.row(i) = Eigen::RowVector3d(0.5, 0.5, 1);
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

// --- hierarchy
    std::vector<std::vector<int>> ancestor_list = find_ancestor_list(V, E_mst, root);
    // color each point with its level, that is the number of ancestors
    // save to a new C_hierarchy
    Eigen::MatrixXd C_hierarchy(V.rows(), 3);
    for (int i = 0; i < V.rows(); ++i) {
        int level = ancestor_list[i].size();
        C_hierarchy.row(i) = generate_distinct_color(level);
    }

// --- components, then dijkstra and mst on each component
    static float component_radius = 2.5f;
    auto component_graphs = get_component_graphs(V, component_radius);

    Eigen::MatrixXd comp_V, comp_C_vertices;
    component_graph_vertices(component_graphs, comp_V, comp_C_vertices);

    Eigen::MatrixXd P1, P2, comp_C_edges;
    component_graph_edges(component_graphs, P1, P2, comp_C_edges);

    std::vector<int> point_to_component_id(V.rows(), -1);
    for (int i = 0; i < component_graphs.size(); ++i) {
        for (int idx : component_graphs[i].global_indices) {
            point_to_component_id[idx] = component_graphs[i].component_id;
        }
    }


// --- viewer
    igl::opengl::glfw::Viewer viewer;
    viewer.core().background_color = Eigen::Vector4f(1.0, 1.0, 1.0, 1.0);  // R, G, B, A

    int index_point_cloud = viewer.append_mesh(); 
    int index_skeleton = viewer.append_mesh();
    int index_selected_points = viewer.append_mesh();
    int index_snake = viewer.append_mesh(); 
    int index_delaunay = viewer.append_mesh(); 
    int index_delaunay_dijkstra = viewer.append_mesh(); 
    int index_mst = viewer.append_mesh(); 
    int index_mst_dijkstra = viewer.append_mesh();
    int index_root = viewer.append_mesh(); 
    int index_hierarchy = viewer.append_mesh(); 
    int index_pointwise_partition = viewer.append_mesh(); 
    int index_segmentwise_partition = viewer.append_mesh(); 
    int index_components = viewer.append_mesh(); 
    int index_components_mst = viewer.append_mesh();



    viewer.data_list[index_point_cloud].point_size = 2; 
    viewer.data_list[index_point_cloud].set_points(V_ori, C_ori); // Original points
    
    viewer.data_list[index_skeleton].point_size = 3; 
    viewer.data_list[index_skeleton].set_points(V, default_C);

    viewer.core().align_camera_center(V);
    viewer.core().camera_eye = Eigen::Vector3f(0, 5, 0); // Set camera position
    viewer.core().camera_up = Eigen::Vector3f(0, 0, 1); 

    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int button, int modifier) {
        return click_point(viewer, V, C, button, modifier, type_colors[current_type_index], type_colors, default_color, index_selected_points, point_to_component_id);
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
    bool show_hierarchy = false;
    bool show_partition = false;
    bool show_components = false;
    bool show_components_mst = false;

    menu.callback_draw_viewer_menu = [&]()
    {
        ImGui::SetNextWindowSize(ImVec2(350, 750), ImGuiCond_FirstUseEver); // width, height
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::Begin("Menu", nullptr, ImGuiWindowFlags_NoCollapse);

        ImGui::Text("Demostration Settings:");
        ImGui::SliderFloat("Skeleton point radius", &viewer.data_list[index_skeleton].point_size, 0.001f, 5.0f);
        ImGui::SliderFloat("Point cloud point radius", &viewer.data_list[index_point_cloud].point_size, 0.001f, 5.0f);


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
            for (int i = 0; i < viewer.data_list.size(); ++i) {
                viewer.data_list[i].clear();
            }
            viewer.data_list[index_point_cloud].set_points(V_ori, C_ori);
            viewer.data_list[index_skeleton].set_points(V, default_C);
            for (int i = 0; i < V.rows(); ++i) {
                C.row(i) = default_color;
            }
            type_labels = {"Type 1"};
            current_type_index = 0;
            type_colors = { generate_distinct_color(current_type_index - 1) };
        }

/*
        ImGui::Separator();
        ImGui::Text("Snake Params:");
        ImGui::SliderInt("snake iteration num", &params.snake_iteration_num, 0, 100);
        ImGui::SliderFloat("snake step", &params.snake_step, 0.0f, 0.2f);
        ImGui::SliderInt("snake resample num", &params.snake_resample_num, 0, 100);
        ImGui::SliderFloat("weight elastic", &params.weight_elastic, 0.0f, 10.0f);
        ImGui::SliderFloat("weight curvature", &params.weight_curvature, 0.0f, 10.0f);
        ImGui::SliderFloat("weight attraction", &params.weight_attraction, 0.0f, 100.0f);
        

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
                viewer.data_list[index_snake].set_edges(V_all, E_all, C_all); 
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
            viewer.data_list[index_snake].clear(); 
        }


*/
        ImGui::Separator();
        ImGui::Text("Delaunay Triangulation:");

        if (ImGui::Button("Delaunay Triangulation", ImVec2(-1, 0))) {
            show_delaunay = !show_delaunay;
            if (show_delaunay) {
                viewer.data_list[index_delaunay].set_edges(V, E_dt_filtered, Eigen::RowVector3d(0.0, 0.0, 0.0));
            }
            else {
                viewer.data_list[index_delaunay].clear(); 
            }
        }

        // get max in dt_edge_lengths
        bool updated_delaunay = ImGui::SliderFloat("max edge length", &filter_edge_length, 0.0f, max_edge_length);
        
        if (updated_delaunay) {
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
                viewer.data_list[index_delaunay].clear();
                viewer.data_list[index_delaunay].set_edges(V, E_dt_filtered, Eigen::RowVector3d(0.0, 0.0, 0.0)); 
            }
            else
            {
                viewer.data_list[index_delaunay].clear(); // clear edges to hide
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
                    viewer.data_list[index_delaunay_dijkstra].set_edges(V, E_all, C_all); 
                }
            }
            else {
                viewer.data_list[index_delaunay_dijkstra].clear(); // clear edges to hide
            }
        }


        ImGui::Separator();

        ImGui::Text("Minimum Spanning Tree:");

        if (ImGui::Button("Minimum Spanning Tree", ImVec2(-1, 0))) {
            show_mst = !show_mst;
            if (show_mst) {
                viewer.data_list[index_mst].set_edges(V, E_mst, Eigen::RowVector3d(0, 0, 0));
            }
            else {
                viewer.data_list[index_mst].clear(); // clear edges to hide
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
                    viewer.data_list[index_mst_dijkstra].set_edges(V, E_all, C_all); 
                }
            }
            else {
                viewer.data_list[index_mst_dijkstra].clear(); // clear edges to hide
            }
        }

        ImGui::Separator();
        ImGui::Text("Tree Hierarchy:");
        if (ImGui::Button("Find Root", ImVec2(-1, 0))) {
            show_root = !show_root;
            if (show_root) {
                viewer.data_list[index_root].set_points(V.row(root), Eigen::RowVector3d(1, 0, 0)); 
                viewer.data_list[index_root].point_size = 15;
            } else {
                viewer.data_list[index_root].clear();
            }   
        }
        if (ImGui::Button("Show Hierarchy Color", ImVec2(-1, 0))) {
            std::cout << "show hierarchy color" << std::endl;
            show_hierarchy = !show_hierarchy;
            if (show_hierarchy) {
                viewer.data_list[index_hierarchy].point_size = 5;
                viewer.data_list[index_hierarchy].set_points(V, C_hierarchy); 
                viewer.data_list[index_skeleton].clear();
            } else {
                viewer.data_list[index_hierarchy].clear(); // clear edges to hide
                viewer.data_list[index_skeleton].set_points(V, default_C);
            }
        }

        ImGui::Separator();
        ImGui::Text("Partition:");

        if (ImGui::Button("Point-wise partition", ImVec2(-1, 0))) {
            show_partition = !show_partition;
            if (show_partition) {
                // 构建 labeled_points: map from point index to class index
                std::unordered_map<int, int> labeled_points;
                for (int class_id = 0; class_id < type_colors.size(); ++class_id) {
                    Eigen::MatrixXd points = get_colored_points(V, C, type_colors[class_id]);
                    for (int i = 0; i < points.rows(); ++i) {
                        int index = find_closest_point(V, points.row(i));
                        labeled_points[index] = class_id;
                    }
                }

                // 调用 partition 函数
                std::unordered_map<int, int> partition = pointwise_partition_with_dijkstra(V, E_mst, labeled_points);

                // 构建颜色矩阵
                Eigen::MatrixXd C_partition_p(V.rows(), 3);
                for (int i = 0; i < V.rows(); ++i) {
                    if (partition.count(i)) {
                        int class_id = partition[i];
                        C_partition_p.row(i) = generate_distinct_color(class_id);
                    } else {
                        C_partition_p.row(i) = default_color; // fallback for未覆盖点（理论上不会有）
                    }
                }
                viewer.data_list[index_pointwise_partition].set_points(V, C_partition_p);
                viewer.data_list[index_pointwise_partition].point_size = 7;
                viewer.data_list[index_pointwise_partition].dirty |= igl::opengl::MeshGL::DIRTY_ALL;
                viewer.data_list[index_skeleton].clear();
            } else {
                viewer.data_list[index_pointwise_partition].clear();
                viewer.data_list[index_skeleton].set_points(V, default_C);
            }
        }



        Eigen::MatrixXd C_debug;

        if (ImGui::Button("Segment-wise partition", ImVec2(-1, 0))) {
            show_partition = !show_partition;
            if (show_partition) {
                // 构建 labeled_points: map from point index to class index
                std::unordered_map<int, int> labeled_points;
                for (int class_id = 0; class_id < type_colors.size(); ++class_id) {
                    Eigen::MatrixXd points = get_colored_points(V, C, type_colors[class_id]);
                    for (int i = 0; i < points.rows(); ++i) {
                        int index = find_closest_point(V, points.row(i));
                        labeled_points[index] = class_id;
                    }
                }

                // 调用 partition 函数
                std::unordered_map<int, int> partition = segment_based_partition_based_on_dijkstra(V, E_mst, labeled_points, &C_debug);

                // 构建颜色矩阵
                Eigen::MatrixXd C_partition(V.rows(), 3);
                for (int i = 0; i < V.rows(); ++i) {
                    if (partition.count(i)) {
                        int class_id = partition[i];
                        C_partition.row(i) = generate_distinct_color(class_id);
                    } else {
                        C_partition.row(i) = default_color; // fallback for未覆盖点（理论上不会有）
                    }
                }
                viewer.data_list[index_segmentwise_partition].set_points(V, C_partition);
                // viewer.data_list[index_pointwise_partition].set_points(V, C_debug);
                viewer.data_list[index_segmentwise_partition].point_size = 7;
                viewer.data_list[index_segmentwise_partition].dirty |= igl::opengl::MeshGL::DIRTY_ALL;
                viewer.data_list[index_skeleton].clear();
            } else {
                viewer.data_list[index_segmentwise_partition].clear();
                viewer.data_list[index_skeleton].set_points(V, default_C);
            }
        }


        ImGui::Separator();
        ImGui::Text("Connected Components:");
        
        bool slider_changed = ImGui::SliderFloat("Component radius", &component_radius, 0.0f, 10.0f);

        static float last_component_radius = component_radius;

        bool update_components = false;
        if (slider_changed && std::abs(component_radius - last_component_radius) >= 0.1f) {
            update_components = true;
            last_component_radius = component_radius;
        }


        if (ImGui::Button("Show Connected Components", ImVec2(-1, 0))) {
            show_components = !show_components;

                if (show_components) {

                        viewer.data_list[index_components].add_points(comp_V, comp_C_vertices);

                        viewer.data_list[index_components].point_size = 5;
                        viewer.data_list[index_skeleton].clear();

                    } else {
                        viewer.data_list[index_components].clear();
                        viewer.data_list[index_skeleton].set_points(V, default_C);
                    }
            }

        if (ImGui::Button("Show MST of Each Components", ImVec2(-1, 0))) {
            show_components_mst = !show_components_mst;

            if (show_components_mst) {

                for (int i = 0; i < P1.rows(); ++i) {
                    viewer.data_list[index_components_mst].add_edges(P1.row(i), P2.row(i), comp_C_edges.row(i));
                }

                viewer.data_list[index_skeleton].clear();

            } else {
                viewer.data_list[index_components_mst].clear();
                viewer.data_list[index_skeleton].set_points(V, default_C);
            }
        }

        if (update_components) {
            viewer.data_list[index_components].clear();
            viewer.data_list[index_components_mst].clear();

            component_graphs = get_component_graphs(V, component_radius);

            for (int i = 0; i < component_graphs.size(); ++i) {
                for (int idx : component_graphs[i].global_indices) {
                    point_to_component_id[idx] = component_graphs[i].component_id;
                }
            }

            component_graph_vertices(component_graphs, comp_V, comp_C_vertices);

            // print radius
            std::cout << "Component radius: " << component_radius << std::endl;

            // print number of components
            std::cout << "Number of components: " << component_graphs.size() << std::endl;

            component_graph_edges(component_graphs, P1, P2, comp_C_edges);

                if (show_components) {

                        viewer.data_list[index_components].add_points(comp_V, comp_C_vertices);

                        viewer.data_list[index_components].point_size = 5;
                        viewer.data_list[index_skeleton].clear();

                    } else {
                        viewer.data_list[index_components].clear();
                        viewer.data_list[index_skeleton].set_points(V, default_C);
                    }
            

                if (show_components_mst) {

                    for (int i = 0; i < P1.rows(); ++i) {
                        viewer.data_list[index_components_mst].add_edges(P1.row(i), P2.row(i), comp_C_edges.row(i));
                    }

                    viewer.data_list[index_skeleton].clear();

                } else {
                    viewer.data_list[index_components_mst].clear();
                    viewer.data_list[index_skeleton].set_points(V, default_C);
                }
update_components = false;
        }



        ImGui::End(); 
    };

    viewer.launch();

    return 0;
}
