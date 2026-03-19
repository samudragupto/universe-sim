find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(GLFW QUIET glfw3)
endif()

if(NOT GLFW_FOUND)
    find_path(GLFW_INCLUDE_DIRS GLFW/glfw3.h)
    find_library(GLFW_LIBRARIES NAMES glfw glfw3)

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(GLFW DEFAULT_MSG GLFW_LIBRARIES GLFW_INCLUDE_DIRS)
endif()