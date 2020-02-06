## Installation on Ubuntu

1. If you haven't already, clone the project by running:
```
git clone --recursive https://github.com/aleksispi/pose-drl.git
```

**Note:** The below steps describing how to setup Caffe and Matcaffe work on Ubuntu 16.04 with Matlab 2018b, but it will likely work for other Ubuntu / Matlab versions as well (although Matlab 2017 seems to have some compatibility issues with Caffe). We have also attached a [guide](INSTALL_2015a.md) specifying how to make it work assuming Ubuntu 16.04 and access to Matlab 2015a. Please note that to use MubyNet yourself (which is needed when you pre-compute poses and deep features) you may have to use Matlab 2015a.

2. At the top-level of our repo, clone the [latest Caffe](https://github.com/BVLC/caffe):
```
git clone https://github.com/BVLC/caffe.git
```

3. Copy `cos_layer.cpp` and `cos_layer.cu` from `install-files` to `caffe/src/caffe/layers`. Copy `cos_layer.hpp` from `install-files` to `caffe/include/caffe/layers/`.

4. Go to `caffe` and create an empty `Makefile.config`. Copy the contents from `Makefile.config.example` into `Makefile.config`.

5. In `Makefile.config`, first uncomment and re-direct the `MATLAB_DIR` to where your Matlab's `bin` folder resides. For example:
```
MATLAB_DIR := /usr/local/share/matlab2018b
```
Then update `CUDA_ARCH` according to your system. The following worked for us:
```
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
	     -gencode arch=compute_35,code=sm_35 \
	     -gencode arch=compute_50,code=sm_50 \
	     -gencode arch=compute_52,code=sm_52 \
	     -gencode arch=compute_60,code=sm_60 \
	     -gencode arch=compute_61,code=sm_61 \
	     -gencode arch=compute_61,code=compute_61

```
Finally, uncomment `WITH_PYTHON_LAYER := 1` and below it set:
```
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial  
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial
```

6. Install caffe as normal by running `make`. See [this link](https://github.com/BVLC/caffe) and/or [this link](https://caffe.berkeleyvision.org/) for help on how to install caffe. Some typical steps on a fresh Ubuntu installation includes running the following prior to `make`:
```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install libatlas-base-dev libopenblas-dev
sudo apt-get install -y libopencv-dev
```

7. Install Matcaffe by running `make matcaffe`. If you encounter any issues, see the troubleshooting section below.

8. In a terminal, go to `code/panoptic/json-matlab/` and type
```
mex json_decode.c jsmn.c
mex json_encode.c

```

9. All done! Return to the [main readme](README.md) for instructions on how to run Pose-DRL, or see this [documentation](dataset/README.md) for how to download and preprocess the Panoptic data, pre-compute MubyNet deep features and pose estimates and train/download instance features for matching.
.

### Troubleshooting for Matcaffe
- If you get mex-crashes, first try exiting Matlab and rerun Matlab from a terminal with the command `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab -desktop` (see also [this link](https://se.mathworks.com/matlabcentral/answers/329796-issue-with-libstdc-so-6)).

- If you still get (some other) mex-crashes than before. try the following commands from your `matlab2015a/bin/glnxa64` folder:

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

- Once done (including running `make clean all matcaffe`), relaunch matlab with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab -desktop`. Then all mex-errors should be gone.
