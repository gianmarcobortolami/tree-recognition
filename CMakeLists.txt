cmake_minimum_required(VERSION 2.8)

project( exam )

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( main src/main.cpp src/TreeDetector.cpp)

target_link_libraries(main ${OpenCV_LIBS})