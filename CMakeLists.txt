cmake_minimum_required(VERSION 3.24)
project(rl_bot_training LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Compatibility for the pinned dependencies")
set(GIGALEARN_SOURCE_DIR "${CMAKE_SOURCE_DIR}/artifacts/gigalearn" CACHE PATH "Pinned GigaLearn checkout")

if(NOT EXISTS "${GIGALEARN_SOURCE_DIR}/GigaLearnCPP/CMakeLists.txt")
  message(FATAL_ERROR "Run scripts/bootstrap-training.ps1 before configuring")
endif()
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${GIGALEARN_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE GIGALEARN_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT GIGALEARN_COMMIT STREQUAL "86e11c3881a21225a43c36b14af40f0b3c87e7dd")
  message(FATAL_ERROR "Unexpected GigaLearn revision: ${GIGALEARN_COMMIT}")
endif()

find_package(Python 3.11 EXACT REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/patch_gigalearn.py"
  --source "${GIGALEARN_SOURCE_DIR}" --check COMMAND_ERROR_IS_FATAL ANY)
# The archived project uses both legacy and modern Python discovery.
set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}" CACHE FILEPATH "" FORCE)
set(PYTHON_INCLUDE_DIR "${Python_INCLUDE_DIRS}" CACHE PATH "" FORCE)
set(PYTHON_LIBRARY "${Python_LIBRARIES}" CACHE FILEPATH "" FORCE)
set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)
find_package(Torch 2.14.0 EXACT REQUIRED)
if(CAFFE2_USE_CUDA)
  message(FATAL_ERROR "Use CPU LibTorch 2.14.0 for this validated build lane")
endif()

if(MSVC)
  add_compile_options(/bigobj /EHsc /wd4251 /wd4275)
endif()
add_subdirectory("${GIGALEARN_SOURCE_DIR}/GigaLearnCPP" "gigalearn")
target_compile_definitions(GigaLearnCPP PUBLIC GGL_STATIC)
# Public GigaLearn headers expose Torch types, although upstream links privately.
target_link_libraries(GigaLearnCPP PUBLIC "${TORCH_LIBRARIES}")

add_executable(rl-bot-train training/main.cpp)
target_include_directories(rl-bot-train PRIVATE training
  "${GIGALEARN_SOURCE_DIR}/GigaLearnCPP/src/private")
target_link_libraries(rl-bot-train PRIVATE GigaLearnCPP)
target_compile_definitions(rl-bot-train PRIVATE
  RLBOT_GIGALEARN_COMMIT="${GIGALEARN_COMMIT}"
  RLBOT_RLGYMCPP_TREE="ccf16f6de81de2e553e9296b7edca7e205ec45c2"
  RLBOT_ROCKETSIM_TREE="43a3765dc8a778f7f4b9168b7db5af94b12767e0"
  RLBOT_PYBIND11_TREE="c277308eee38ec457032f338301bb21e766339a8"
  RLBOT_TORCH_VERSION="${Torch_VERSION}"
  RLBOT_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
  RLBOT_BUILD_TYPE="$<CONFIG>")

enable_testing()
add_test(NAME trainer-help COMMAND rl-bot-train --help)
find_package(Threads REQUIRED)
add_executable(threadpool-check tests/threadpool_check.cpp)
target_include_directories(threadpool-check PRIVATE
  "${GIGALEARN_SOURCE_DIR}/GigaLearnCPP/RLGymCPP/src"
  "${GIGALEARN_SOURCE_DIR}/GigaLearnCPP/RLGymCPP/thread_pool")
target_link_libraries(threadpool-check PRIVATE Threads::Threads)
add_test(NAME worker-exception-propagation COMMAND threadpool-check)
set_tests_properties(trainer-help worker-exception-propagation PROPERTIES TIMEOUT 30)
