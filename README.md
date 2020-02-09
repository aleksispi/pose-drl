# Deep Reinforcement Learning for Active Human Pose Estimation

**Authors:** [Aleksis Pirinen](https://aleksispi.github.io/)\*, [Erik Gärtner](https://gartner.io/)\*, and [Cristian Sminchisescu](http://www.maths.lth.se/sminchisescu/) _(* denotes first authorship)_.

![visualization](https://user-images.githubusercontent.com/32370520/70310219-68cc0a80-180f-11ea-9726-169942378db7.png)

## Overview
Official implementation of the AAAI 2020 paper _Deep Reinforcement Learning for Active Human Pose Estimation_.
This repository contains code for producing results for our Pose-DRL model and the baselines that match those reported in the paper, as well as training Pose-DRL on Panoptic. Our paper is on [arXiv](https://arxiv.org/abs/2001.02024) too. A video overview of this paper is available [here](https://www.youtube.com/watch?v=CNsTbX_q4so), with step-by-step visualizations [here](https://www.youtube.com/watch?v=9hB6aZCBMbs) and [here](https://www.youtube.com/watch?v=tIOmNUnemNw).

Pose-DRL is implemented in [Caffe](https://github.com/BVLC/caffe). The experiments are performed in the [CMU Panoptic](http://domedb.perception.cs.cmu.edu/) multi-camera framework.
The Pose-DRL model in this repository uses [MubyNet](http://www.maths.lth.se/sminchisescu/media/papers/integrated-3d-sensing-of-multiple-people-in-natural-images_neurips2018.pdf) as underlying 3d human pose estimator. 

![overview](https://user-images.githubusercontent.com/32370520/70310282-8ac58d00-180f-11ea-8115-fd9451932538.png)

### Citation
If you find this implementation and/or our paper interesting or helpful, please consider citing:

    @inproceedings{gartner2020activehpe,
      title={Deep Reinforcement Learning for Active Human Pose Estimation},
      author={G\"{a}rtner, Erik and Pirinen, Aleksis and Sminchisescu, Cristian},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      year={2020}
    }

## Setup
1. Clone the repository.
2. Read the following [documentation](INSTALL.md) on how to setup our system, assuming you are using pre-computed feature maps and pose predictions for MubyNet (see the next item on how to pre-compute these). This covers prerequisites and how to install our framework.
3. See this [dataset documentation](dataset/README.md) for how to download and preprocess the Panoptic data, pre-compute MubyNet deep features and pose estimates and train/download instance features for matching.

## Pretrained models
Pretrained model weights for Pose-DRL can be downloaded [here](https://lu.box.com/s/bvxcbj99e1ei5gy17mznjjrwefatqcex).

## Using Pose-DRL

### Training the model
To train the model run the command:
```
run_train_agent('train')
```

The results and weights will be stored in the location of `CONFIG.output_dir`.

### Evaluating the model
Given the model weights (either the provided weights or your own):

1. Set flag `CONFIG.evaluation_mode = 'test';`
2. Set flag `CONFIG.agent_random_init = 0;`
3. Set flag `CONFIG.agent_weights = '<your-weights-path>';`
4. Set flag `CONFIG.training_agent_nbr_eps = 1;` (Note, this will not update weights, since they are updated every 40 eps.)
5. Run `run_train_agent('train');`, results will be stored in the location of `CONFIG.output_dir`.

### Acknowledgements
This work was supported by the European Research Council Consolidator grant SEED, CNCS-UEFISCDI PN-III-P4-ID-PCE-2016-0535, the EU Horizon 2020 Grant DE-ENIGMA, SSF, as well as the Wallenberg AI, Autonomous Systems and Software Program ([WASP](https://wasp-sweden.org/)) funded by the Knut and Alice Wallenberg Foundation. Finally, we would like to thank [Alin Popa](https://alinionutpopa.github.io/), [Andrei Zanfir](https://scholar.google.ro/citations?user=8lmzWycAAAAJ&hl=en), [Mihai Zanfir](https://scholar.google.com/citations?user=af68sKkAAAAJ&hl=en) and [Elisabeta Oneata](https://scholar.google.com/citations?user=ssHUPeUAAAAJ&hl=en) for helpful discussions and support.
