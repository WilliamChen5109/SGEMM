cmake_minimum_required(VERSION 3.16.3)

project(Sgemm C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

# 使用 CMake 的 FindOpenMP 模块
find_package(OpenMP REQUIRED)

# 添加编译选项（保留原有的优化选项）
add_compile_options(-O3 -pthread -mavx512f)

# 如果使用 OpenMP，添加 OpenMP 编译标志
if(OpenMP_C_FOUND)
    add_compile_options(${OpenMP_C_FLAGS})
endif()

include_directories(./inc)

# 查找BLIS库 - 使用你指定的路径
set(BLIS_INCLUDE_DIR "/home/kun/spack/opt/spack/linux-x86_64_v4/amdblis-5.2-7c4ickkpixrejjmg4j7x44csmdh4ucma/include/blis")
set(BLIS_LIBRARY_DIR "/home/kun/spack/opt/spack/linux-x86_64_v4/amdblis-5.2-7c4ickkpixrejjmg4j7x44csmdh4ucma/lib")

# 添加BLIS头文件路径
include_directories(${BLIS_INCLUDE_DIR})

# 查找Threads库（pthread）
find_package(Threads REQUIRED)

set(MAIN_SOURCES
    src/Main.c
)

set(NATIVE_SOURCES
    src/native.c
)

set(OPT_SOURCES
    src/optimization.c
)

set(TEST_SGEMM_SOURCES
    src/test_sgemm.c
)

set(ALL_SOURCES
    ${MAIN_SOURCES}
    ${NATIVE_SOURCES}
    ${OPT_SOURCES}
    ${TEST_SGEMM_SOURCES}
)

add_executable(MyTest ${ALL_SOURCES})

target_link_libraries(MyTest 
    PRIVATE
    Threads::Threads           # pthread库
    OpenMP::OpenMP_C           # OpenMP 库
    ${BLIS_LIBRARY_DIR}/libblis.so
    m                          # 数学库
)

# 或者，如果上面的 OpenMP::OpenMP_C 不工作，可以手动添加
# target_link_libraries(MyTest PRIVATE -fopenmp)