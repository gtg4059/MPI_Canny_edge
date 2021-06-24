cmake_minimum_required(VERSION 3.17)
project(mpi_exam)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXE_LINKER_FLAGS "-static")
add_executable(canny main.cpp EasyBMP.cpp)

include_directories("Include")
include_directories("Include/x64")
find_library(
        Test_Lib
        NAMES msmpi.lib msmpifec.lib msmpifmc.lib
        HINTS ./lib
        REQUIRED)
target_link_libraries(canny LINK_PUBLIC ${Test_Lib})
