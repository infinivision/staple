cmake_minimum_required(VERSION 2.8)
project(stapleCC)

IF (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3") #c++11
ENDIF (UNIX)

#IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
#  set(OpenCV_DIR /home/firefly/wanglei/opencv_build)
#  include_directories(${OpenCV_INCLUDE_DIRS})
#endif ()

if (APPLE)
  include_directories("/Users/load/code/cpp/eigen-git-mirror-3.3.4")
elseif (UNIX)
  include_directories("/home/firefly/wanglei/eigen-git-mirror-3.3.4")
endif ()

if (APPLE)
  include_directories( "/usr/local/include" )
  link_directories("/usr/local/lib")
elseif (UNIX)
  include_directories( "/usr/include" )
  link_directories("/usr/lib")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif ()

add_subdirectory("./staple")
include_directories("./staple")
add_executable(${PROJECT_NAME} main.cpp)

find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} trackerStaple ${OpenCV_LIBS} fftw3f)

add_executable(bench bench.cpp)
target_link_libraries(bench trackerStaple ${OpenCV_LIBS} fftw3f)

add_executable(export export.cpp)
target_link_libraries(export fftw3f)