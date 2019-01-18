# IndoorGeoNet

This code is the implementation of the following paper in Pytorch:

DeepPBM: DEEP PROBABILISTIC BACKGROUND MODEL ESTIMATIONFROM VIDEO SEQUENCES

Behnaz Rezaei, Amirreza Farnoosh and Sarah Ostadabbas


## Requirements

This code is tested on Python3.6, Pytorch 1.0 and CUDA 8.0 on Ubuntu 16.04. MATLAB R2016b.

## Data preparation

The following dataset is used for experiments in the paper:

BMC2012 dataset:

```
@inproceedings{vacavant2012benchmark,
  title={A benchmark dataset for outdoor foreground/background extraction},
  author={Vacavant, Antoine and Chateau, Thierry and Wilhelm, Alexis and Lequi{\`e}vre, Laurent},
  booktitle={Asian Conference on Computer Vision},
  pages={291--300},
  year={2012},
  organization={Springer}
}
```

After downloading the dataset, you should run BMC2012DataLoader.py to preprocess dataset and get .npy files.

## Training and Testing

You should run BetaVAE_BMC2012_Vid#.py files for training the network for each specicfic video of BMC2012 dataset, and generating background images for each frame. 

### Foreground mask generation

You should run MaskExtraction_BMC2012.m to generate binary foreground masks from generated background images from the previous steps.

### Quantitative results

You should run processVideoFolder.m , and then confusionMatrixToVar.m to generate quantitative results. 

## Reference
