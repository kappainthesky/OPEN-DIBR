cmake_minimum_required(VERSION 3.7)
project(GStreamerReceiverPoC CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${GST_INCLUDE_DIRS}
    ${GST_APP_INCLUDE_DIRS}
)

add_executable(gstreamer_receiver receiver.cpp)

target_link_libraries(gstreamer_receiver PRIVATE
    PkgConfig::GST
    PkgConfig::GST_APP
)
