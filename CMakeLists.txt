cmake_minimum_required(VERSION 3.24)
project(universe-sim LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

add_subdirectory(third_party/glfw)
set(GLAD_SOURCES third_party/glad/src/glad.c)

set(CUDA_SOURCES
    src/cuda/kernels.cu
    src/cuda/interop.cu
    src/cuda/morton.cu
    src/cuda/octree.cu
    src/cuda/physics.cu
    src/cuda/stellar_evolution.cu
    src/cuda/trails.cu
    src/cuda/density_field.cu
    src/cuda/diagnostics.cu
)

set(CPP_SOURCES
    src/main.cpp
    src/application.cpp
    src/render/renderer.cpp
    src/render/shader_manager.cpp
    src/render/post_process.cpp
    src/render/camera.cpp
    src/render/text_renderer.cpp
    src/render/screenshot.cpp
    src/render/volumetric_renderer.cpp
    src/physics/simulation.cpp
    src/physics/particle_system.cpp
    src/physics/initial_conditions.cpp
    src/ui/input_handler.cpp
)

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES} ${GLAD_SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86;89;90"
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/glad/include
    ${CMAKE_SOURCE_DIR}/third_party/glm
    ${CMAKE_SOURCE_DIR}/third_party/stb
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    OpenGL::GL 
    glfw 
    CUDA::cudart 
    CUDA::curand
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 --use_fast_math -Xcompiler="/Zc:preprocessor">
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C>:/O2>
)

file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/config DESTINATION ${CMAKE_BINARY_DIR})