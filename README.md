# Vehicle Tracking using Lucas Kanade Forward Alignment Method

### Terminology
**LKT**: Lukas Kanade Tracker

**Momentum**: Averaging the transformation matrix over the previous frames

### Demos
#### Expected output on executing the code
[Demo video](https://youtu.be/Kp99f36Dc38)

#### Result gifs

Momentum + LKT with Affine Transform (C++ and Python implementations)

![Momentum_LK_affine Demo](results/Momentum_LK_affine.gif)

Momentum + LKT with Translation-only Transform (Python-only implementation)

![Momentum_LK_translation Demo](results/Momentum_LK_translation.gif)

Momentum + Inverse Baker Method (Python-only implementation)

![Momentum_MB Demo](results/Momentum_MB.gif)

### Dependencies
- OpenCV
- Eigen

### Build Steps
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

### Future work
- [ ] Assumes the availability of `groundtruth.txt` for calculation of IoU. Add a flag that allows us to run on unlabelled data.
- [ ] Implement occlusion detection function to remove dependence on `occlusion.label` groundtruth.
- [ ] Perform LKT over features rather than pixels

### Author(s)

* **Siddharth Saha** - [trunc8](https://github.com/trunc8)

<p align='center'>Created with :heart: by <a href="https://www.linkedin.com/in/sahasiddharth611/">Siddharth</a></p>
