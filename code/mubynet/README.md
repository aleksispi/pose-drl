# MubyNet
This package contains code for the [Deep Network for the Integrated 3D Sensing of Multiple People in Natural Images](https://papers.nips.cc/paper/8061-deep-network-for-the-integrated-3d-sensing-of-multiple-people-in-natural-images.pdf) (MubyNet) paper published at the NeurIPS 2018 from Montreal, Canada.

![architecture](fig/methodOverview.png)
![encoding](fig/3d_encoding.png)

By using the software, you are agreeing to the terms of the [license agreement](https://github.com/alinionutpopa/MubyNet/blob/master/LICENSE).

Our software is built on top of the [Caffe](http://caffe.berkeleyvision.org/) deep learning library. The current version was developed by:

[Andrei Zanfir](https://scholar.google.ro/citations?user=8lmzWycAAAAJ&hl=en&oi=ao)
,
[Elisabeta Marinoiu](https://scholar.google.ro/citations?user=ssHUPeUAAAAJ&hl=en&oi=ao)
,
[Mihai Zanfir](https://scholar.google.ro/citations?user=af68sKkAAAAJ&hl=en&oi=ao)
,
[Alin-Ionut Popa](https://scholar.google.ro/citations?user=HmFCNrsAAAAJ&hl=en&oi=ao)
and
[Cristian Sminchisescu](http://www.maths.lth.se/matematiklth/personal/sminchis/index.html)

Our model identifies human body structures (joints and limbs) in images, groups them based on 2d and 3d information fused using learned scoring functions, and optimally aggregates such responses into partial or complete 3d human skeleton hypotheses under kinematic tree constraints, but without knowing in advance the number of people in the scene and their visibility relations. We design a multi-task deep neural network with differentiable stages where the person grouping problem is formulated as an integer program based on learned body part scores parameterized by both 2d and 3d information. This avoids suboptimality resulting from separate 2d and 3d reasoning, with grouping performed based on the combined representation.

![sample1](fig/19_full.png)
![sample2](fig/59_full.png)

If you use this code/model for your research, please cite the following paper:
```
@inproceedings{mubynet_2018,
    author = {Zanfir, Andrei and Marinoiu, Elisabeta and Zanfir, Mihai and Popa, Alin-Ionut and Sminchisescu, Cristian},
    title  = {Deep Network for the Integrated 3D Sensing of Multiple People in Natural Images},
    booktitle = {Advances in Neural Information Processing Systems 31},
    year   = {2018}
}

```


## Installation Guide
First, clone the project by running:
```
git clone --recursive https://github.com/alinionutpopa/MubyNet.git
```

You need to compile the Caffe library in this repository. See http://caffe.berkeleyvision.org/installation.html for the latest installation instructions.

Check the users group in case you need help:
https://groups.google.com/forum/#!forum/caffe-users


All done! Try our method!


### 1.3 Run the demo
First download the model that includes the trained weights from this [link](https://drive.google.com/open?id=1bYgGzr-ha1mjT_uNLIMKb-ExcBHRVN5x) into the ``data/model`` folder. Also, change the ``caffepath`` variable from ``code/config_release.m`` file accordingly.

The MATLAB script for running the demo is ``demoMubyNet.m``. Please note that the shape estimation module will be made available at a later time.

Contact: <andrei.zanfir@imar.ro>, <elisabeta.marinoiu@imar.ro>, <mihai.zanfir@imar.ro>, <alin.popa@imar.ro>, <cristian.sminchisescu@math.lth.se>


### Acknowledgments
This work was supported in part by the European Research Council Consolidator grant SEED, CNCS-UEFISCDI (PN-III-P4-ID-PCE-2016-0535, PN-III-P4-ID-PCCF-2016-0180), the EU Horizon 2020 grant DE-ENIGMA (688835), and SSF.
