cmake_minimum_required(VERSION 3.16)
project(example)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set module path
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# set(LIBIGL_WITH_KNN ON)


# Libigl
include(libigl)

# Ensure NanoFlann Include Path is Correct
include_directories($ENV{HOME}/nanoflann/include)
include_directories(${CMAKE_SOURCE_DIR}/src)


# Enable the target igl::glfw
igl_include(glfw)
# Other modules you could enable
#igl_include(embree)
igl_include(imgui)
# igl_include(knn)
#igl_include(opengl)
# igl_include(stb)
#igl_include(predicates)
#igl_include(xml)
#igl_include(copyleft cgal)
#igl_include(copyleft comiso)
#igl_include(copyleft core)
#igl_include(copyleft cork)
#igl_include(copyleft tetgen)
#igl_include(restricted matlab)
#igl_include(restricted mosek)
#igl_include(restricted triangle)

# file(GLOB SRC_FILES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/*.cpp)

# Add your project files
# file(GLOB SRC_FILES *.cpp)
# add_executable(${PROJECT_NAME} ${SRC_FILES})
# add_executable( 
#     main.cpp
#     connect_points/nearest_neighbor.cpp
# )
include_directories(src)
# Collect source files
set(SRC_FILES 
    src/main.cpp
    src/connect_points/nearest_neighbor.cpp
    src/connect_points/travel_salesman.cpp
    src/connect_points/snake.cpp
)

# Add the executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Link igl (and the glfw module) to your project
target_link_libraries(${PROJECT_NAME} PUBLIC 
  igl::glfw
  igl::core
  # igl::knn
  ## Other modules you could link to
  # igl::embree
  igl::imgui
  # igl::opengl
  # igl::stb
  # igl::predicates
  # igl::xml
  # igl_copyleft::cgal
  # igl_copyleft::comiso
  # igl_copyleft::core
  # igl_copyleft::cork
  # igl_copyleft::tetgen
  # igl_restricted::matlab
  # igl_restricted::mosek
  # igl_restricted::triangle
  )
