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
#include "connect_points/nearest_neighbor.h"
#include "connect_points/travel_salesman.h"
#include "connect_points/snake.h"
#include "params.h"



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
        if (C.row(vid) == Eigen::RowVector3d(1.0, 1.0, 1.0)) {
            C.row(vid) = selected_color;
        } 
        else {
            C.row(vid) = Eigen::RowVector3d(1.0, 1.0, 1.0);
        }
        viewer.data().set_points(V, C);
        return true;
    }

    return false;
}



std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> connect_point(GUIParams& params, const Eigen::MatrixXd& V, const Eigen::MatrixXd& C, const Eigen::RowVector3d& color) {
    std::vector<int> selected_points;

    for (int i = 0; i < V.rows(); i++) {
        if (C.row(i) == color) {
            selected_points.push_back(i);
        }
    }

    if (selected_points.size() < 2) {
        // return Eigen::MatrixXi(); // Return an empty matrix
        return std::make_tuple(Eigen::MatrixXd(), Eigen::MatrixXi());
    }



    Eigen::MatrixXd points(selected_points.size(), 3);

    for (int i = 0; i < selected_points.size(); i++) {
        points.row(i) = V.row(selected_points[i]);
    }

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
        std::tie(V_, E_) = snake(params, V, V_raw.row(E_raw(i, 0)), V_raw.row(E_raw(i, 1)));
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
    Eigen::MatrixXd V;
    std::string filename = "/Users/ruox/Documents/DoubleDegree/cours_2/ParcoursRecherche/projet/generate_example_point_cloud/point_cloud/X_form_C.xyz";
    if (!loadXYZ(filename, V)) return 1;
    Eigen::MatrixXd C = Eigen::MatrixXd::Constant(V.rows(), 3, 1.0);
    Eigen::MatrixXi E;

    igl::opengl::glfw::Viewer viewer;

    //menu
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

    GUIParams params; 

    std::vector<std::string> type_labels = {"Type 1"};
    std::vector<Eigen::RowVector3d> type_colors = {
        Eigen::RowVector3d::Random().cwiseAbs()  // Type 1 color
    };
    int current_type_index = 0;  // Initially selected type

    menu.callback_draw_viewer_menu = [&]()
    {
        ImGui::SetNextWindowSize(ImVec2(350, 350), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::Begin("Menu", nullptr, ImGuiWindowFlags_NoCollapse);

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
            type_colors.push_back(Eigen::RowVector3d::Random().cwiseAbs());  // New random color
            current_type_index = new_index - 1;  // Select the newly added type
        }

        ImGui::Separator();

        if (ImGui::Button("connect", ImVec2(-1, 0)))
        {
            std::vector<Eigen::RowVector3d> all_colors;
            Eigen::MatrixXi E_all(0, 2);
            Eigen::MatrixXd V_all(0, 3);


            for (int i = 0; i < type_labels.size(); i++) {
                Eigen::MatrixXd V_temp = V;
                Eigen::MatrixXd C_temp = C;
                auto [V_result, E_result] = connect_point(params, V_temp, C_temp, type_colors[i]);

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

                viewer.data().set_edges(V_all, E_all, C_all); 
            }
        }


        ImGui::End(); 
    };

    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int button, int modifier) {
        return click_point(viewer, V, C, button, modifier, type_colors[current_type_index]);
    };
    
    viewer.data().set_points(V, C);

    viewer.launch();

    return 0;
}
