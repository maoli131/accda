# Active Continuous Continual Domain Adaptation

This repo combines the [multi-anchor active domain adaptation method](https://github.com/munanning/MADA) with [continous, continual domain adaptation method]((https://arxiv.org/abs/2009.12518)) into a novel method: ACCDA. With the help of active sampling strategy, ACCDA helps generalizes neural networks trained on source dataset to unseen target datasets with continously changing visual conditions.  

## Data

The **data** directory (only containing file paths) is meant to store the original versions of the [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/), 
[SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/) and [CITYSCAPES](https://www.cityscapes-dataset.com/) datasets. These datasets are available online, and are not included in this repository.

## Running the code

The datasets need to be prepared before running the E2E model. This is done via the **process_gta5_cityscapes.ipynb** or **process_synthia_cityscapes.ipynb**
notebooks. The processed images will be available in the **processed-data** folder.

After loading the data and processing it, run **feat-anchor-active-samples.ipynb** to generate source/target features, clustered anchors and a list of active samples to select. 

After data loading and sample selection, running E2E training and adaptation can be done by running
**vgg16-deeplabv3-GTA5-CITYSCAPES.ipynb** or **vgg16-deeplabv3-SYNTHIA-CITYSCAPES.ipynb**

The notebooks will save model weights in the **weights** folder. Currently, this folder comes prepopulated with weights corresponding 
to the runs present in the notebooks. 

## Original Codebases

The entire codebase is adapted from ["Unsupervised Model Adaptation for Continual Semantic Segmentation"](https://arxiv.org/abs/2009.12518)
The code was tested using Tensorflow 2.2 and CUDA 11.2, with driver version 460.xx. 
```
@misc{stan2021unsupervised,
      title={Unsupervised Model Adaptation for Continual Semantic Segmentation}, 
      author={Serban Stan and Mohammad Rostami},
      year={2021},
      eprint={2009.12518},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

The MADA method for active domain adaptation is adapted from the PyTorch [codebase](https://github.com/munanning/MADA).
```
@inproceedings{ning2021multi,
  title={Multi-Anchor Active Domain Adaptation for Semantic Segmentation},
  author={Ning, Munan and Lu, Donghuan and Wei, Dong and Bian, Cheng and Yuan, Chenglang and Yu, Shuang and Ma, Kai and Zheng, Yefeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9112--9122},
  year={2021}
}
```
