# mini-caffe.cmake

option(USE_MKLDNN "Use mkldnn support" ON)
# select BLAS
set(BLAS "openblas" CACHE STRING "Selected BLAS library")

if(USE_MKLDNN)
  add_definitions(-DUSE_MKLDNN)
endif()

# turn on C++11
if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# include and library
# third party header path
if(MSVC)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/include
                      ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/openblas
                      ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/google
                      ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/mkldnn
                      ${CMAKE_CURRENT_LIST_DIR}/include)
  link_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/lib
                      ${CMAKE_CURRENT_LIST_DIR}/3rdparty/windows/lib)
  list(APPEND Caffe_LINKER_LIBS debug libprotobufd optimized libprotobuf
                                libopenblas mkldnn)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else(MSVC)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
  include_directories(3rdparty/linux/include/mkldnn/)
  list(APPEND Caffe_LINKER_LIBS protobuf)
  if(BLAS STREQUAL "openblas")
    list(APPEND Caffe_LINKER_LIBS openblas)
    message(STATUS "Use OpenBLAS for blas library")
  else()
    list(APPEND Caffe_LINKER_LIBS blas)
    message(STATUS "Use BLAS for blas library")
  endif()
  file(GLOB MKLDNN_LIB 3rdparty/linux/lib/*.so)
  list(APPEND Caffe_LINKER_LIBS ${MKLDNN_LIB})
endif(MSVC)

# source file structure
file(GLOB CAFFE_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include/caffe/*.h
                        ${CMAKE_CURRENT_LIST_DIR}/include/caffe/*.hpp)
file(GLOB CAFFE_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp
                    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
file(GLOB CAFFE_SRC_LAYERS ${CMAKE_CURRENT_LIST_DIR}/src/layers/*.hpp
                           ${CMAKE_CURRENT_LIST_DIR}/src/layers/*.cpp)
file(GLOB CAFFE_SRC_LAYERS_INTEL ${CMAKE_CURRENT_LIST_DIR}/src/layers/intel/*.hpp
                           ${CMAKE_CURRENT_LIST_DIR}/src/layers/intel/*.cpp)
file(GLOB CAFFE_SRC_UTIL ${CMAKE_CURRENT_LIST_DIR}/src/util/*.hpp
                         ${CMAKE_CURRENT_LIST_DIR}/src/util/*.cpp)
file(GLOB CAFFE_SRC_PROTO ${CMAKE_CURRENT_LIST_DIR}/src/proto/caffe.pb.h
                          ${CMAKE_CURRENT_LIST_DIR}/src/proto/caffe.pb.cc)

# cpp code
set(CAFFE_COMPILE_CODE ${CAFFE_INCLUDE}
                       ${CAFFE_SRC}
                       ${CAFFE_SRC_LAYERS}
                       ${CAFFE_SRC_LAYERS_INTEL}
                       ${CAFFE_SRC_UTIL}
                       ${CAFFE_SRC_PROTO})

# file structure
source_group(include FILES ${CAFFE_INCLUDE})
source_group(src FILES ${CAFFE_SRC})
source_group(src\\layers FILES ${CAFFE_SRC_LAYERS})
source_group(src\\util FILES ${CAFFE_SRC_UTIL})
source_group(src\\proto FILES ${CAFFE_SRC_PROTO})

add_definitions(-DCAFFE_EXPORTS)
add_library(caffe SHARED ${CAFFE_COMPILE_CODE})
target_link_libraries(caffe ${Caffe_LINKER_LIBS})
