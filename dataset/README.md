# Dataset

This readme contains the instructions for downloading, extracting and cleaning the dataset used in our Pose-DRL paper. It also provides the steps to compute pose predictions, deep features and instance features required for our Pose-DRL agent. All pose predictions are pre-computed to speed-up RL training.

Main requirement is Matlab2015a and a CUDA-enabled GPU for computing MubyNet features and feature maps (see details below). If you do not want to compute these features yourself (e.g. if you have got them from elsewhere), then most versions of Matlab should be ok (i.e. training the Pose-DRL view selection policy does not in itself require a particular Matlab version). You also need Python 3.6 for downloading and pre-processing the Panoptic data.

## Dataset structure

Pose-DRL assumes that the dataset and the associated cache of pose predictions are provided using a certain structure.

There are two important folders:
- The _data_ folder that contains the raw images and Panoptic annotations (used for evaluating the 3d error).
- The _cache_ folder that contains the MubyNet predictions, deep features and instance features used for matching.

Internally these two folders mirror each other. They assume a second level of folders that separate the scenes into train, test and validation splits.

Example:
```
data/
 train/
   scene1/
 val/
   scene2/
 test/
   scene3/

cache/
 train/
   scene1/
 val/
   scene2/
 test/
   scene3/
```

## Creating the dataset

#### 0. Create folder structure

You may split the `data/` and `cache/` folder into different drives.

1. Create your data folder (e.g. `data/`) somewhere. Do _not_ create any subfolders.
2. Create your cache folder (e.g. `cache/`) somewhere. Then create `cache/test`, `cache/train`, `cache/val`.
3. Go into the `load_config.m` file and set the `dataset_path` and `dataset_cache` flags to the corresponding paths (they should point to `data/` and `cache/`, respectively).

#### 1. Download and clean Panoptic data

1. Download the scenes and extract annotations and images:
    - `./downloadAll.sh <data-folder-path>`
2. Clean-up the scene to remove bad camera frames and ensure that each scene has a fixed number of persons:
    - `python3 clean_up_panoptic.py --check-same-nbr-frames  --check-green-frame --hd --same-people --split-scene --min-con 100 --delete --min-nbr-people 1 <data-folder-path>`
3. Split the data into the official train, validation and test split by running:
    - `bash split_data.sh <data-folder-path>`.

#### 2 Download Faster R-CNN detections

1. Download the pre-computed [train detections](https://lu.box.com/s/w2ckkzsj9aetcmue3jopi1ehk63hjiaj), [val detections](https://lu.box.com/s/op3aq42cn9xgmizpcrq2tnr5c9ao6xpq) and [test detections](https://lu.box.com/s/gaax5sxxelytxqrp37wf96ynytnhb0hg) (zip files).
2. Unzip the associated zip-files.
3. Move all the `detections.h5` files to the top-level of the respective scene folders of the Panoptic data you downloaded in step "1. Download and clean Panoptic data" above.

Note: You may of course use any detector you want and compute detections yourself. Just make sure to save each scene's detections into a h5-file in the format matching the given pre-computed ones.

#### 3. Compute MubyNet predictions and deep features

1. Set `CONFIG.predictor_disabled = 0;` and `CONFIG.predictor_precache = 0;` in `load_config.m`.
2. Download [weights](https://lu.box.com/s/aok4qebxrivy28am3qnqgwusnnzkgw3k) to `models/mubynet/`.
3. For each split (train, val, test) compute the 3d pose predictions and deep features.
    - `run_generate_mubynet_cache('train')`
    - `run_generate_mubynet_cache('val')`
    - `run_generate_mubynet_cache('test')`

#### 4. Compute instance features for matching

Next we want to generate the instance features used for matching people in the scene by appearance. The model is first trained for 40k iterations on the training split, then fine-tuned 2k iterations for each individual scene.

The base of the instance features comes from a VGG-19 model.

1. Download the VGG-19 [weights](https://lu.box.com/s/eswxcfj9gn2qhjp8o7o6mxrihcgrvkpx) (sha1: 7e1441c412647bebdf7ae9750c0c9aba131a1601).
2. Either run the 40k base training on the _train_ split or download the weights.
    - Train weights from scratch using the Matlab script: `run_train_instance_detector('train')`
    - Download pre-trained 40k iteration [weights](https://lu.box.com/s/4dofbrcyz6yr9tmnmvdg62beoa8vbx1l) (sha1: 6727771807b0984f2f3bbed2cf4e0a2af80b396f).
3. Generate the fine-tuned weights for each split:
    - `run_generate_finetuned_instance_cache('<path-40k-instance-weights>', 'train', '<path-to-vgg19-weights>', 2000)`
    - `run_generate_finetuned_instance_cache('<path-40k-instance-weights>', 'val', '<path-to-vgg19-weights>', 2000)`
    - `run_generate_finetuned_instance_cache('<path-40k-instance-weights>', 'test', '<path-to-vgg19-weights>', 2000)`

## Additional information

Below follows in-depth information about the dataset and the scripts.

#### Explanation of scripts
  - `downloadAll.sh` downloads, extracts and, verifies all scenes.
  - `downloadScene.sh` downloads a scene with all videos and annoations
  - `extractScene.sh` extract images from the videos, removes videos and extracts the annotations. The videos frames are subsampled by a provided frequency. The annotations are then pruned to match the frames. Finally any Coco19 annotations are converted to MPII for all scenes, if they exists.
  - `subsample.sh` removes all but every n:th file in a directory.
  - `vgaImgsExtractor.sh` / `hdImgsExtractor.sh` extract image frames from the video then calls subsample on the resulting frames.
  - `verifyScene.sh` checks the content of the dataset.
  - `clean_up_dataset.py` removes bad frames and frames missing annotations
  - `discard_annotations.py` removes annotations to match the subsampled frames.
  - `coco2mpii.py` converts the coco19 annotations to the MPII 15 joint format.

## Acknowledgements

  - Panoptic dataset scripts adapted from https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox
  - We received MubyNet code directly from the authors. Thanks!
