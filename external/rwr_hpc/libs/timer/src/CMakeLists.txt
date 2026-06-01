# Gather source files for current timer method(s)
set(TIMER_SRC
  Timer.cpp
)

# Define the timer library target
add_library(timer ${TIMER_SRC})

target_include_directories(timer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)
