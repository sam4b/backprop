include(FetchContent)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(backend "backprop.hpp" "Backpropagation.cpp" "GPUNetwork.cu" "Layer.hpp"
"Network.hpp")


FetchContent_Declare(
    json
    URL "https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.zip"
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)

FetchContent_MakeAvailable(eigen)

target_link_libraries(backend
  PUBLIC
  rapidcsv nlohmann_json::nlohmann_json Eigen3::Eigen)