# cpp
add_executable(run_net ${CMAKE_CURRENT_LIST_DIR}/run_net.cpp)

if(MSVC)
    target_link_libraries(run_net caffe)
else(MSVC)
    target_link_libraries(run_net caffe pthread)
endif(MSVC)
