### Steps
```sh
# Check if you already have the dependencies using
pkg-config --modversion opencv4
pkg-config --modversion eigen3

# Otherwise install using
sudo apt install libopencv-dev
sudo apt install libeigen3-dev

mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release

cmake --build .

################## Optional Argument flags ######################
# viz_input: To visualize the input frames
# viz_groundtruth: To visualize the groundtruth bounding boxes
# viz_groundtruth_on_tracking: To overlay groundtruth bboxes on tracking
#################################################################
../bin/main viz_input viz_groundtruth viz_groundtruth_on_tracking

# You should be inside build directory to run gtests
ctest
```

TODO:

- [ ] Assumes the availability of `groundtruth.txt` for calculation of IoU. Add a flag that allows us to run on unlabelled data.
- [ ] Implement occlusion detection function to remove dependence on `occlusion.label` groundtruth.
