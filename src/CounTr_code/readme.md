# CounTr: A Novel End-to-End Transformer Approach for Single Image Crowd Counting 

## Outline

Modeling context information is critical for single image crowd counting. Current prevailing fully-convolutional network (FCN) based crowd counting methods cannot effectively capture long-range dependencies with limited receptive fields. Although recent efforts on inserting dilated convolutions and attention modules have been taken to enlarge the receptive fields, the FCN architecture remains unchanged and retains the fundamental limitation on learning long-range relationships. To tackle the problem, we introduce CounTr, a novel end-to-end transformer approach for single image crowd counting, which enables capture global context in every layer of the Transformer. To be specific, CounTr is composed of a powerful transformer-based hierarchical encoder-decoder architecture. The transformer-based encoder is directly applied to sequences of image patches and outputs multi-scale features. The proposed hierarchical self-attention decoder fuses the features from different layers and aggregates both local and global context features representations. 

### Prerequisites

Python 3.8.3. and the following packages are required to run the scripts:

- [PyTorch-1.7.1 and torchvision](https://pytorch.org)  

- Package [timm](https://github.com/rwightman/pytorch-image-models)
		  [fastai](https://github.com/fastai/fastai)
		  [albumentations](https://github.com/albumentations-team/albumentations)
		  [opencv](https://github.com/opencv/opencv-python)


- Dataset: please download the dataset and put images into the folder data/[name of the dataset]/

- Pre-Trained Weights: using swin pretrained weights for encoder/[swin_base_patch4_window7_224.pth] 

To clone the environment, please use the following code with provided requirements.txt file

conda install --name countr --file requirements.txt

### Code Structure

There are three parts in the code:
 - model: 
 			- ounTR.py 
 				contains counTR model codes

 - data: 
 			- crowd_dataset.py 
 				The crowd counting dataset codes, the Base dataset is the abstract of the building block. The Crop_Dataset inherits from Base dataset and used for training and inferencing
 			- transforms.py
 				Transformations that used for model training

 - main.py: the main file to train and evaluate the model.
 - train_loop.py: Fitter class that wraps the training and validation loop
 - utilis.py: helpper function goes here


### Main Hyper-parameters

We introduce the usual hyper-parameters as below. There are some other hyper-parameters in the code, which are only added to make the code general, but not used for experiments in the paper.

#### Basic Parameters

- `path`: root path of dataset

- `train-image-folder`: folder name: folder that contains all train images

- `train-dmap-folder`: folder name: folder that contains all density maps of training image

- `valid-image-folder`: folder name: folder that contains all validation images

- `valid-dmap-folder`: folder name: folder that contains all density maps of validation image

- `output-folder`: folder name: folder that will be used to store weights and training logs

- `ext`: image extension, default `.jpg`

- 'dmap-ext': density map extension, default `.npy`

#### Optimization Parameters

- `epochs`: The maximum number of epochs to train the model, default to `300`

- `lr`: The learning rate, default to `0.0001`

- `init_weights`: The path to the init weights

- `batch-size`: The number of inputs for each batch, default to `4`

- `crop_size`: The designed input size to preprocess the image, default to `448`

- `log-para`: scaler for smoothing training, default 1000



#### Other Parameters

- `device`: To select which GPU device to use, default to `cuda:0`.

### Demonstrations

Usage:
$ python3 main.py --input-path --train-image-folder --train-dmap-folder --valid-image-folder --valid-dmap-folder --output-folder