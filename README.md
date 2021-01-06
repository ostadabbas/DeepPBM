# DeepPBM: Deep Probabilistic Background Modeling

This code is the implementation of the following paper accepted to the ICPR2020 Workshop on Deep Learning for Pattern Recognition (DLPR20):

DeepPBM: Deep Probabilistic Background Model Estimation from Video Sequences (https://arxiv.org/pdf/1902.00820.pdf)

Authors: Amirreza Farnoosh, Behnaz Rezaei, and Sarah Ostadabbas
Corresponding Author: ostadabbas@ece.neu.edu


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

@article{farnoosh2020deeppbm,
  title={DeepPBM: deep probabilistic background model estimation from video sequences},
  author={Farnoosh, Amirreza and Rezaei, Behnaz and Ostadabbas, Sarah},
  journal={The Third International Workshop on Deep Learning for Pattern Recognition (DLPR20), in conjunction with the 25th International Conference on Pattern Recognition (ICPR 2020)},
  year={2020}
}

## For further inquiry please contact: 
Sarah Ostadabbas, PhD
Electrical & Computer Engineering Department
Northeastern University, Boston, MA 02115
Office Phone: 617-373-4992
ostadabbas@ece.neu.edu
Augmented Cognition Lab (ACLab) Webpage: http://www.northeastern.edu/ostadabbas/
