# VLA architecture registry.
#
# VLA_ARCH_KEYS lists every supported architecture (also the GGUF arch string
# and the ROBORT_MODELS selector).  For each key <key> the two variables below
# describe the implementation:
#   VLA_ARCH_<key>_SRC -- implementation source, relative to src/.
#   VLA_ARCH_<key>_DEF -- macro defined on the vla library when this arch is
#                         built (used by api/model.cpp to wire dispatch).
#
# src/CMakeLists.txt reads this table generically: selecting ROBORT_MODELS=<key>
# (or building all by default) compiles exactly the registered sources, so
# adding a new architecture does not require touching the build logic.
#
# To add a new architecture:
#   1. add its key to VLA_ARCH_KEYS and the two VLA_ARCH_<key>_* variables
#      below,
#   2. create the implementation under src/models/<arch>/ defining
#      std::unique_ptr<vla::Model> <arch>_create(mmproj, ckpt, config),
#   3. add the matching per-arch block in src/api/model.cpp (factory
#      declaration, detection string, dispatch case) -- all three are grouped
#      and marked with "ARCH:" comments.
set(VLA_ARCH_KEYS pi05 hy_vla)

set(VLA_ARCH_pi05_SRC   models/pi05/pi05.cpp)
set(VLA_ARCH_pi05_DEF   VLA_HAS_PI05)

set(VLA_ARCH_hy_vla_SRC models/hy_vla/hy_vla.cpp)
set(VLA_ARCH_hy_vla_DEF VLA_HAS_HY_VLA)
