INCLUDE_DIR=-I/opt/homebrew/Caskroom/miniforge/base/include/
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup $(python3 -m pybind11  --includes) ${INCLUDE_DIR} $1.cpp -o $1$(python3-config --extension-suffix)