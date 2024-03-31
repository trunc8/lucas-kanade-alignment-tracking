### Steps
```sh
sudo apt install libopencv-dev
sudo apt install libeigen3-dev

# Check that you have the dependency using
pkg-config --modversion opencv4
pkg-config --modversion eigen3

mkdir build && cd build

cmake ..

cmake --build .

../bin/main
```
