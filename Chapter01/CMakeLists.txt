cmake_minimum_required(VERSION 3.14)
project(deeplearn)

set(CMAKE_PREFIX_PATH D:/lib/libtorch-win-shared-with-deps-1.5.1+cpu/libtorch)
set(OpenCV_DIR D:/opencv/build/x64/vc15/lib)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(deeplearn  main.cpp)
target_link_libraries(deeplearn ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET deeplearn PROPERTY CXX_STANDARD 11)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET deeplearn
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:deeplearn>)

    file(GLOB OPENCV_DLLS "D:/opencv/build/x64/vc15/bin/*.dll")
    add_custom_command(TARGET deeplearn
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS}
            $<TARGET_FILE_DIR:deeplearn>)
endif (MSVC)