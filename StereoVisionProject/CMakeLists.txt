# 1. 声明 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 2. 设置项目名称
project(StereoVision)

# 3. 指定使用 C++ 11 标准
set(CMAKE_CXX_STANDARD 11)

# 4. 寻找 OpenCV 库 
find_package(OpenCV REQUIRED)

# 5. 指定生成的可执行文件名，以及它包含哪些源代码文件
add_executable(stereo_app main.cpp)

# 6. 将 OpenCV 的头文件和库连接到这个程序上
target_link_libraries(stereo_app ${OpenCV_LIBS})
