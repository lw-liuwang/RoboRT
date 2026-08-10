# src: RoboRT pi0.5 VLA inference engine + serving.
#
# Layout is by responsibility:
#   api/     public C++ API (vla::Config / Inputs / predict / last_stats)
#   models/  model implementations (pi05/), one directory per architecture
#   serving/ ZeroMQ/Protobuf server, observation adapter, protobuf schema

find_package(Threads REQUIRED)

set(TARGET vla-pi05)
add_library(${TARGET}
            api/model.h
            models/pi05/pi05.cpp)

set_target_properties(${TARGET} PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
)

target_link_libraries     (${TARGET} PUBLIC mtmd)
target_link_libraries     (${TARGET} PRIVATE Threads::Threads)
target_include_directories(${TARGET} PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/api)
target_compile_features   (${TARGET} PRIVATE cxx_std_17)

if (GGML_CUDA)
    target_link_libraries(${TARGET} PUBLIC ggml-cuda)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# self-check: dump one VLA_PI05_SEED=42 action chunk so runs can be compared
# bit-exact against a reference (e.g. the RoboRT wrapper engine).
set(TARGET vla-pi05-selfcheck)
add_executable(${TARGET} models/pi05/selfcheck.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME vla-pi05-selfcheck)
if(ROBORT_INSTALL_TOOLS)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries  (${TARGET} PRIVATE vla-pi05 Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# vla-pi05-server: ZeroMQ/Protobuf serving for the pi0.5 engine.
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

set(TARGET vla-pi05-server)
add_executable(${TARGET}
    serving/server.cpp
    serving/server_base.cpp
    serving/adapter.cpp
    ${VLA_PROTO_GEN_DIR}/vla.pb.cc
)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME vla-pi05-server)
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
    vla-pi05
    Threads::Threads
    ${Protobuf_LIBRARIES}
    PkgConfig::ZeroMQ
)
