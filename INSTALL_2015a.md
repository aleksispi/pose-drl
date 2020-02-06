## Installation on Ubuntu 16.04 with access to Matlab 2015a

1. If you haven't already, clone the project by running:
```
git clone --recursive https://github.com/aleksispi/pose-drl.git
```

2. Make sure you have Matlab 2015a installed on your system. Then follow the steps [here](INSTALL.md). Note that on a fresh Ubuntu 16.04, it may be necessary to follow a few steps from https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide as well (in particular, installing some more packages than the ones suggested in the [DMHS repo](https://github.com/alinionutpopa/dmhs)).

In general install:
```bash
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install libatlas-base-dev libopenblas-dev
sudo apt-get install -y libopencv-dev
```

Then compile caffe using

```bash
make
make matcaffe
```

3. If you get mex-crashes, first try exiting matlab and rerun matlab from command window with the command ‘LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab -desktop’ (see also https://se.mathworks.com/matlabcentral/answers/329796-issue-with-libstdc-so-6).

4. You may now get other mex-crashes than before. If you get such new mex-crashes, run the following commands from your `matlab2015a/bin/glnxa64` folder.

```bash
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so libopencv_calib3d.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_contrib.so libopencv_contrib.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_core.so libopencv_core.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_features2d.so libopencv_features2d.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_flann.so libopencv_flann.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_gpu.so libopencv_gpu.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_highgui.so libopencv_highgui.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so libopencv_imgproc.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_legacy.so libopencv_legacy.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_ml.so libopencv_ml.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so libopencv_objdetect.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_photo.so libopencv_photo.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_stitching.so libopencv_stitching.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_video.so libopencv_video.so.2.4
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libopencv_videostab.so libopencv_videostab.so.2.4
```

5. Once done (including running ‘make clean all matcaffe’), relaunch matlab with ‘LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab -desktop’. Then all mex-errors should be gone.

6. In a terminal, go to `code/panoptic/json-matlab/` and type
```
mex json_decode.c jsmn.c
mex json_encode.c

```

### Installing without GPU-support (or with CUDA 9.0)
Our repo is runnable on CUDA 9.0, but it's a non-official and hacky approach which involves giving up GPU-support for some (currently unused) layers. But the following also explains how to make it work in CPU-mode (which `matcaffe` does not officially support during installation -- it assumes you have CUDA installed). Running in CPU-mode yields (for the 3fc-models) a 70% increase in runtime per episode (110% increase in runtime per agent forward pass).

1. Begin by following steps 1 - 2 in the standard instructions above, and stop right before compiling caffe.

2. Edit `CUDA_ARCH` in `Makefile.config`, setting it as follows:
```bash
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_50,code=compute_50
```

3. Temporarily move the files
```
caffe/src/caffe/layers/softmax_layer.cu
caffe/src/caffe/util/math_functions.cu
```
outside the `caffe` path. Note that this will imply that you cannot use `softmax_layer` nor `math_functions` in GPU-mode, but for running our models it still works. Everything will be runnable in CPU-mode in the end however. If you want, you can move these two files back into the original path once done with installation (just remember to move them each time you want to run `make`).

4. Continue with the compilation of caffe and matcaffe and follow the remaining steps in the standard installation instructions.
