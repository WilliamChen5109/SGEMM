cmake_minimum_required(VERSION 3.16.3)

project(Sgemm C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

add_compile_options(-O3 -Wall -Wextra)

include_directories(./inc)

set(MAIN_SOURCES
    src/Main.c
)

set(NATIVE_SOURCES
    src/native.c
)

set(TEST_SGEMM_SOURCES
    src/test_sgemm.c
)

set(ALL_SOURCES
    ${MAIN_SOURCES}
    ${NATIVE_SOURCES}
    ${TEST_SGEMM_SOURCES}
)

add_executable(MyTest ${ALL_SOURCES})