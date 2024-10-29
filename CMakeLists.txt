cmake_minimum_required(VERSION 3.11)
project(tokenizer)


set(SOURCE_FILES main.cpp)
include_directories(./include())
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

list(APPEND SOURCE_FILES ${SOURCES})
message(STATUS "source_path="${SOURCE_FILES})

add_executable(tokenizer ${SOURCE_FILES})

