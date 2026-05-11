include(FetchContent)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(backend "backprop.hpp"  "GPUNetwork.cu" "Layer.hpp"
"Network.hpp" "Utilities.hpp")


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


FetchContent_Declare(
    rapidcsv
    URL "https://github.com/d99kris/rapidcsv/archive/v8.90.zip"
)
FetchContent_MakeAvailable(rapidcsv)


find_package(CUDAToolkit REQUIRED)
target_include_directories(backend PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(backend
  PUBLIC
  nlohmann_json::nlohmann_json Eigen3::Eigen CUDA::cudart CUDA::cublas rapidcsv)