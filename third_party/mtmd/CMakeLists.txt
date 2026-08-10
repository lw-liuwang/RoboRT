# mtmd: vision encoding library (upstream llama.cpp, trimmed to RoboRT needs)
#
# Only the clip* vision-tower machinery is kept: clip.cpp (graph builder +
# self-contained GGUF loader), mtmd-image.cpp (clip_image_* pre/postprocess
# helpers it calls), and the vision model builders under models/.
# The llama.cpp LLM text tower (mtmd.cpp / mtmd-helper.cpp / mtmd-audio.cpp)
# was removed — RoboRT's pi0.5 engine only uses clip_* APIs.

find_package(Threads REQUIRED)

add_library(mtmd
            clip.cpp
            clip.h
            clip-impl.h
            clip-model.h
            clip-graph.h
            mtmd-image.cpp
            mtmd-image.h
            models/models.h
            models/cogvlm.cpp
            models/conformer.cpp
            models/dotsocr.cpp
            models/gemma4a.cpp
            models/gemma4v.cpp
            models/glm4v.cpp
            models/hunyuanocr.cpp
            models/internvl.cpp
            models/kimivl.cpp
            models/kimik25.cpp
            models/nemotron-v2-vl.cpp
            models/llama4.cpp
            models/llava.cpp
            models/minicpmv.cpp
            models/paddleocr.cpp
            models/pixtral.cpp
            models/qwen2vl.cpp
            models/qwen3vl.cpp
            models/qwen3a.cpp
            models/step3vl.cpp
            models/siglip.cpp
            models/whisper-enc.cpp
            models/deepseekocr.cpp
            models/mobilenetv5.cpp
            models/youtuvl.cpp
            models/yasa2.cpp
            )

set_target_properties(mtmd PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_link_libraries     (mtmd PUBLIC ggml)
target_link_libraries     (mtmd PRIVATE Threads::Threads)
target_include_directories(mtmd PUBLIC  .)
target_compile_features   (mtmd PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties     (mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

install(TARGETS mtmd LIBRARY)

if (NOT MSVC)
    target_compile_options(mtmd PRIVATE -Wno-cast-qual)
endif()

if (ANDROID)
    # miniaudio.h defines ma_android_sdk_version() without a prior prototype
    target_compile_options(mtmd PRIVATE -Wno-missing-prototypes)
endif()

if (TARGET BUILD_INFO)
    add_dependencies(mtmd BUILD_INFO)
endif()
