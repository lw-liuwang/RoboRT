# src: RoboRT robot-brain inference engine + serving.
#
# Layout is by responsibility:
#   api/     public C++ API (robo:: / vla:: types, predict, last_stats)
#            policy.h  -- new robo:: namespace (aliases to vla:: for compat)
#            model.h   -- legacy vla:: namespace (backward compatible)
#   models/  model implementations, one directory per architecture
#   serving/ ZeroMQ/Protobuf server, observation adapter, protobuf schema

find_package(Threads REQUIRED)

# VLA architectures to build, space-separated (e.g. "pi05 hy_vla").
# Empty (default) = build every architecture in src/models/architectures.cmake.
# Each selected arch defines a VLA_HAS_<ARCH> macro on the library; the
# dispatcher in api/model.cpp wires only the compiled architectures, and the
# selection is data-driven from src/models/architectures.cmake so new models
# only require a row there.
set(ROBORT_MODELS "" CACHE STRING
    "VLA architectures to build, space-separated (empty = all in src/models/architectures.cmake)")

include(models/architectures.cmake)

# Resolve selection: empty -> all registered architectures.  ROBORT_MODELS may
# use spaces or semicolons as separators (a command-line -D value arrives as a
# single space-separated string, so split it explicitly).
string(REPLACE " " ";" VLA_SELECT "${ROBORT_MODELS}")
if(NOT VLA_SELECT)
    set(VLA_SELECT ${VLA_ARCH_KEYS})
endif()

set(VLA_MODEL_SOURCES "")
set(VLA_MODEL_DEFS "")
foreach(VLA_ARCH IN LISTS VLA_SELECT)
    if(NOT DEFINED VLA_ARCH_${VLA_ARCH}_SRC)
        message(FATAL_ERROR "Unknown ROBORT_MODELS entry '${VLA_ARCH}' (registered: ${VLA_ARCH_KEYS})")
    endif()
    list(APPEND VLA_MODEL_SOURCES ${VLA_ARCH_${VLA_ARCH}_SRC})
    list(APPEND VLA_MODEL_DEFS     ${VLA_ARCH_${VLA_ARCH}_DEF})
endforeach()
list(REMOVE_DUPLICATES VLA_MODEL_SOURCES)
list(REMOVE_DUPLICATES VLA_MODEL_DEFS)
list(REMOVE_DUPLICATES VLA_SELECT)
if(NOT VLA_MODEL_SOURCES)
    message(FATAL_ERROR "ROBORT_MODELS resolved to no architectures (registered: ${VLA_ARCH_KEYS})")
endif()
message(STATUS "RoboRT architectures to build: ${VLA_SELECT}")

set(TARGET robort)
add_library(${TARGET}
            api/model.h
            api/policy.h
            api/model.cpp
            ${VLA_MODEL_SOURCES})

set_target_properties(${TARGET} PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
)

target_link_libraries     (${TARGET} PUBLIC mtmd)
target_link_libraries     (${TARGET} PRIVATE Threads::Threads)
target_compile_definitions(${TARGET} PRIVATE ${VLA_MODEL_DEFS})
target_include_directories(${TARGET} PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/api)
target_compile_features   (${TARGET} PRIVATE cxx_std_17)

if (GGML_CUDA)
    target_link_libraries(${TARGET} PUBLIC ggml-cuda)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Provide legacy alias so any external cmake consumers using vla-pi05 still work.
add_library(vla-pi05 ALIAS robort)

# self-check: dump one seed=42 action chunk for bit-exact regression testing.
set(TARGET robort-selfcheck)
add_executable(${TARGET} models/pi05/selfcheck.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME robort-selfcheck)
if(ROBORT_INSTALL_TOOLS)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries  (${TARGET} PRIVATE robort Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# lingbot_vla_v2 alignment driver: replays reference inputs (precomputed clip
# embeddings / tokens / state / noise) and dumps every tapped intermediate for
# numerical comparison against export_lingbot_vla_v2_reference.py.
set(TARGET robort-lingbot-v2-selfcheck)
add_executable(${TARGET} models/lingbot_vla_v2/selfcheck.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME robort-lingbot-v2-selfcheck)
if(ROBORT_INSTALL_TOOLS)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries  (${TARGET} PRIVATE robort Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# robort-server: ZeroMQ/Protobuf serving for the RoboRT engine.
if (ROBORT_BUILD_SERVER)
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED IMPORTED_TARGET libzmq)

set(VLA_PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/serving/vla.proto)
set(VLA_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen)
file(MAKE_DIRECTORY ${VLA_PROTO_GEN_DIR})

add_custom_command(
    OUTPUT  ${VLA_PROTO_GEN_DIR}/vla.pb.h
            ${VLA_PROTO_GEN_DIR}/vla.pb.cc
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/serving
            --cpp_out=${VLA_PROTO_GEN_DIR}
            ${VLA_PROTO_SRC}
    DEPENDS ${VLA_PROTO_SRC}
    VERBATIM
)

set(TARGET robort-server)
add_executable(${TARGET}
    serving/server.cpp
    serving/server_base.cpp
    serving/adapter.cpp
    ${VLA_PROTO_GEN_DIR}/vla.pb.cc
)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME robort-server)
if(ROBORT_INSTALL_TOOLS)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/serving
    ${CMAKE_CURRENT_SOURCE_DIR}/api
    ${VLA_PROTO_GEN_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/stb
    ${Protobuf_INCLUDE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    robort
    Threads::Threads
    ${Protobuf_LIBRARIES}
    PkgConfig::ZeroMQ
)
endif()
