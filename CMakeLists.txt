# RoboRT: real-time vision-language-action (VLA) inference engine for pi0.5.
#
# Vendored upstream llama.cpp provides the underlying compute stack (ggml
# graph + CUDA backend under third_party/ggml, mtmd vision tower under
# third_party/mtmd). This project keeps only the pieces RoboRT depends on and
# builds them together with the RoboRT engine (src/).
# See README.md for the upstream reference.

cmake_minimum_required(VERSION 3.14...3.28)
project("RoboRT" C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_WARN_UNUSED_CLI YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_STANDALONE ON)
    include(git-vars)
else()
    set(LLAMA_STANDALONE OFF)
endif()

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

if (MINGW)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)
else()
    set(BUILD_SHARED_LIBS_DEFAULT ON)
endif()
option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

option(LLAMA_USE_SYSTEM_GGML        "use system libggml instead of building ggml"     OFF)
option(ROBORT_INSTALL_TOOLS         "install RoboRT tools"                            ${LLAMA_STANDALONE})

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)

if (NOT DEFINED LLAMA_BUILD_NUMBER)
    set(LLAMA_BUILD_NUMBER        ${BUILD_NUMBER})
endif()
if (NOT DEFINED LLAMA_BUILD_COMMIT)
    set(LLAMA_BUILD_COMMIT        ${BUILD_COMMIT})
endif()
set(LLAMA_INSTALL_VERSION 0.0.${LLAMA_BUILD_NUMBER})

include("cmake/license.cmake")
license_add_file("RoboRT" "LICENSE")

# 3rd-party: ggml compute library (upstream llama.cpp)

if (LLAMA_USE_SYSTEM_GGML)
    message(STATUS "Using system-provided libggml, skipping ggml build")
    find_package(ggml REQUIRED)
    add_library(ggml ALIAS ggml::ggml)
endif()

if (NOT TARGET ggml AND NOT LLAMA_USE_SYSTEM_GGML)
    set(GGML_BUILD_NUMBER ${LLAMA_BUILD_NUMBER})
    set(GGML_BUILD_COMMIT ${LLAMA_BUILD_COMMIT})
endif()

# third_party: vendored upstream code (ggml compute core, mtmd vision tower)
add_subdirectory(third_party)

# src: RoboRT pi0.5 engine / server / convert scripts
add_subdirectory(src)