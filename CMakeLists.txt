cmake_minimum_required (VERSION 3.13)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/Modules)

include (CheckIPOSupported)

project (siphon LANGUAGES CXX)

if (NOT MSVC)
    if (NOT CMAKE_BUILD_TYPE)
        set (CMAKE_BUILD_TYPE "Release")
    endif ()
    set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

find_program (CCACHE ccache)
if (CCACHE)
    set_property (GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE})
    set_property (GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE})
endif ()

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")

find_package (OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS            "${CMAKE_CXX_FLAGS}             ${OpenMP_CXX_FLAGS}")
    set (CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS}   ${OpenMP_STATIC_LINKER_FLAGS}")
    set (CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS}   ${OpenMP_SHARED_LINKER_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS     "${CMAKE_EXE_LINKER_FLAGS}      ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

option (USE_LTO "Enable LTO." ON)
if (USE_LTO)
    check_ipo_supported (RESULT CMAKE_CHECK_IPO_RESULT OUTPUT CMAKE_CHECK_IPO_OUTPUT)
    if (CMAKE_CHECK_IPO_RESULT)
        message(STATUS "Found CMake IPO support")
    else ()
        message(WARNING "IPO is not supported: ${CMAKE_CHECK_IPO_OUTPUT}")
        set (USE_LTO OFF)
    endif ()
endif ()

option (USE_NATIVE_ARCH "Optimize for host's architecture only without backward compatibility." ON)
include (CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG ("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE AND USE_NATIVE_ARCH)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
else ()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
endif ()

file (GLOB_RECURSE SIPHON_SOURCES "${PROJECT_SOURCE_DIR}/src/siphon/*.cpp")
add_library (siphon SHARED ${SIPHON_SOURCE})
set_target_properties (siphon PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${USE_LTO})

find_package (OpenCV REQUIRED)
if (OpenCV_FOUND)
    target_include_directories (siphon PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries (siphon PUBLIC ${OpenCV_LIBS})
endif ()

find_package (Threads REQUIRED)
target_link_libraries (siphon PUBLIC ${CMAKE_THREAD_LIBS_INIT})

find_package (Caffe2 REQUIRED)
if (CAFFE2_FOUND)
    target_link_libraries (siphon PUBLIC ${Caffe2_MAIN_LIBS})
endif ()

file (GLOB_RECURSE APP_SOURCES "${PROJECT_SOURCE_DIR}/src/app/*.cpp")
foreach (filename ${APP_SOURCES})
    get_filename_component (name ${filename} NAME_WE)
    add_executable (${name} ${filename})
    set_target_properties (${name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${USE_LTO})
    target_link_libraries (${name} siphon)
endforeach ()

