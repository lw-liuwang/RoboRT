# third_party: vendored dependencies.
#
# Everything here is upstream code that RoboRT builds against but does not
# own. ggml (compute core) and mtmd (vision tower) have CMake build steps;
# gguf-py (GGUF writer for the convert scripts) and stb (stb_image.h JPEG
# decode for the server) are consumed without a build step.

if (NOT LLAMA_USE_SYSTEM_GGML)
    add_subdirectory(ggml)
endif()

add_subdirectory(mtmd)
