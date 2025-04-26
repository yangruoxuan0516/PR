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

#define CONNECT_METHOD 1 // 0 for shortest path
                         // 1 for snake
                         // default for greedy

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

bool click_point(igl::opengl::glfw::Viewer& viewer, Eigen::MatrixXd& V, Eigen::MatrixXd& C, int, int) {
    int vid = -1;
    double min_dis = 20;
    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y - 1;        
    Eigen::Vector3f pos;
    pos << x, y, 0;

    for (int i = 0; i < V.rows(); i++){
        Eigen::Vector3f vertex_screen;
        igl::project(V.row(i).cast<float>(), viewer.core().view, viewer.core().proj, viewer.core().viewport, vertex_screen);

        double dis = (vertex_screen.head<2>() - pos.head<2>()).norm();
        if (dis < min_dis){
            min_dis = dis;
            vid = i;
        }
    }

    if (vid != -1) {
        if (C.row(vid) == Eigen::RowVector3d(1, 0, 0))
            C.row(vid) << 1, 1, 1;
        else 
            C.row(vid) << 1, 0, 0;

        viewer.data().set_points(V, C);
        C.resize(V.rows(), 3);
        return true;
    }
    return false;
}



std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> connect_point(GUIParams& params, const Eigen::MatrixXd& V, const Eigen::MatrixXd& C) {
    std::vector<int> selected_points;

    for (int i = 0; i < V.rows(); i++) {
        if (C.row(i) == Eigen::RowVector3d(1, 0, 0)) {
            selected_points.push_back(i);
        }
    }

    if (selected_points.size() < 2) {
        // return Eigen::MatrixXi(); // Return an empty matrix
        return std::make_tuple(Eigen::MatrixXd(), Eigen::MatrixXi());
    }

    // Minimum Hamiltonian Path - Open Path

    int n = selected_points.size();
    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0));

    // Compute pairwise distances
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                // dist[i][j] = euclidean_distance(V.row(selected_points[i]), V.row(selected_points[j]));
                dist[i][j] = (V.row(selected_points[i]) - V.row(selected_points[j])).norm();
            }
        }
    }


    switch (params.connect_method)
    {
        case 0:{
            std::cout << "\nSnake" << std::endl;

            Eigen::MatrixXd points(selected_points.size(), 3);

            for (int i = 0; i < selected_points.size(); i++) {
                points.row(i) = V.row(selected_points[i]);
            }

            Eigen::MatrixXd V_;
            Eigen::MatrixXi E_;
            
            std::tie(V_, E_) = snake(params, V, points.row(0), points.row(1));

            return std::make_tuple(V_,E_);
            break;
        }

        case 1:
        {
            std::cout << "\nSalesman" << std::endl;

            Eigen::MatrixXd points(selected_points.size(), 3);

            for (int i = 0; i < selected_points.size(); i++) {
                points.row(i) = V.row(selected_points[i]);
            }

            Eigen::MatrixXd V;
            Eigen::MatrixXi E;

            std::tie(V, E) = travel_salesman(points);

            return std::make_tuple(V,E);
            break;
        }

        default:
        { 
            std::cout << "\nGreedy" << std::endl;

            Eigen::MatrixXd points(selected_points.size(), 3);

            for (int i = 0; i < selected_points.size(); i++) {
                points.row(i) = V.row(selected_points[i]);
            }

            Eigen::MatrixXd V;
            Eigen::MatrixXi E;

            std::tie(V, E) = nearest_neighbor(points);

            return std::make_tuple(V,E);
            break;
        }
    }
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

    menu.callback_draw_viewer_menu = [&]()
    {
        ImGui::SetNextWindowSize(ImVec2(350, 250), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::Begin("Menu", nullptr, ImGuiWindowFlags_NoCollapse);

        ImGui::SliderInt("connect method", &params.connect_method, 0, 2);
        ImGui::SliderInt("snake iteration num", &params.snake_iteration_num, 0, 100);
        ImGui::SliderFloat("snake step", &params.snake_step, 0.0f, 0.2f);
        ImGui::SliderInt("snake resample num", &params.snake_resample_num, 0, 100);
        ImGui::SliderFloat("weight elastic", &params.weight_elastic, 0.0f, 10.0f);
        ImGui::SliderFloat("weight curvature", &params.weight_curvature, 0.0f, 10.0f);
        ImGui::SliderFloat("weight attraction", &params.weight_attraction, 0.0f, 100.0f);


        if (ImGui::Button("connect", ImVec2(-1, 0)))
        {
            Eigen::MatrixXd V_copy = V;
            auto [V, E] = connect_point(params, V_copy, C);
            if (E.rows() > 0) {
                viewer.data().set_edges(V, E, Eigen::RowVector3d(1, 0.5, 0.5));
            }
        }
        ImGui::End(); 
    };
    

    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int button, int modifier) {
        return click_point(viewer, V, C, button, modifier);
    };
    viewer.data().set_points(V, C);

    viewer.launch();

    return 0;
}
