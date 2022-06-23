import voxelmorph2d as vm2d
import voxelmorph3d as vm3d
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
from skimage.transform import resize
import skimage
import multiprocessing as mp
from tqdm import tqdm
import gc
import time
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
use_gpu = torch.cuda.is_available()
print('is there cuda?', use_gpu)

class VoxelMorph():
    """
    VoxelMorph Class is a higher level interface for both 2D and 3D
    Voxelmorph classes. It makes training easier and is scalable.
    """

    def __init__(self, input_dims, is_2d=False, use_gpu=True):
        self.dims = input_dims
        if is_2d:
            self.vm = vm2d
            self.voxelmorph = vm2d.VoxelMorph2d(input_dims[0] * 2, use_gpu)
        else:
            self.vm = vm3d
            self.voxelmorph = vm3d.VoxelMorph3d(input_dims[0] * 2, use_gpu)
        self.optimizer = optim.SGD(
            self.voxelmorph.parameters(), lr=1e-4, momentum=0.99)
        self.params = {'batch_size': 1,
                       'shuffle': True,
                       #'num_workers': 6,
                       'worker_init_fn': np.random.seed(42)
                       }
        self.device = torch.device("cuda:0" if use_gpu else "cpu")

    def check_dims(self, x):
        try:
            if x.shape[1:] == self.dims:
                return
            else:
                raise TypeError
        except TypeError as e:
            print("Invalid Dimension Error. The supposed dimension is ",
                  self.dims, "But the dimension of the input is ", x.shape[1:])

    def forward(self, x):
        self.check_dims(x)
        return voxelmorph(x)

    def calculate_loss(self, y, ytrue, n=9, lamda=0.01, is_training=True):
        loss = torch.abs(self.vm.vox_morph_loss(y, ytrue, n, lamda)) + torch.abs(self.vm.smooothing_loss(y))
        return loss

    def train_model(self, batch_moving, batch_fixed, n=9, lamda=0.01, return_metric_score=True):
        self.optimizer.zero_grad()
        batch_fixed, batch_moving = batch_fixed.to(
            self.device), batch_moving.to(self.device)
        registered_image = self.voxelmorph(batch_moving, batch_fixed)
        train_loss = torch.abs(self.calculate_loss(
            registered_image, batch_fixed, n, lamda))
        train_loss.backward()
        self.optimizer.step()
        if return_metric_score:
            train_dice_score = self.vm.dice_score(
                registered_image, batch_fixed)
            return train_loss, train_dice_score
        return train_loss

    def get_test_loss(self, batch_moving, batch_fixed, n=9, lamda=0.01):
        with torch.set_grad_enabled(False):
            registered_image = self.voxelmorph(batch_moving, batch_fixed)

            val_loss = torch.abs(self.vm.vox_morph_loss(
                registered_image, batch_fixed, n, lamda)) + torch.abs(self.vm.smooothing_loss(registered_image))
            val_dice_score = self.vm.dice_score(registered_image, batch_fixed)
            return val_loss, val_dice_score


def vxm_data_generator(x_data, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        moving_images = torch.tensor(moving_images)
        fixed_images = torch.tensor(fixed_images)

        inputs = moving_images
        
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = fixed_images
        
        yield (inputs, outputs)

class Dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, batch_size):
        'Initialization'
        self.x_data = data
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if use_gpu else "cpu")

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        vol_shape = self.x_data.shape[1:] # extract data shape
        ndims = len(vol_shape)
        
        # prepare a zero array the size of the deformation
        # we'll explain this below
        zero_phi = np.zeros([self.batch_size, *vol_shape, ndims])
        
            # prepare inputs:
            # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, self.x_data.shape[0], size=1)
        moving_images = resize(self.x_data[idx1, ..., np.newaxis],(256,256,1))
        #moving_images = skimage.color.gray2rgb(moving_images)
        #print('increase scale',moving_images.shape)
        moving_images = torch.tensor(moving_images).reshape(1,256,256,1)
        idx2 = np.random.randint(0, self.x_data.shape[0], size=1)
        fixed_images = resize(self.x_data[idx2, ..., np.newaxis], (256,256,1))
        #fixed_images = skimage.color.gray2rgb(fixed_images)
        #print('increase fixed scale',fixed_images.shape)
        fixed_images = torch.tensor(fixed_images).reshape(1,256,256,1)
            #inputs = [moving_images, fixed_images]
            
            #outputs = [fixed_images, zero_phi]
            
        return fixed_images, moving_images


def main():
    '''
    In this I'll take example of FIRE: Fundus Image Registration Dataset
    to demostrate the working of the API.
    '''
    vm = VoxelMorph(
        (1, 256, 256), is_2d=True)  # Object of the higher level class
    DATA_PATH = '/content/drive/MyDrive/BRAIN_REGISTER/tutorial_data.npz'
    params = {'batch_size': 1,
              'shuffle': True,
              #'num_workers': 6,
              'worker_init_fn': np.random.seed(42)
              }
    print('Got the model')
    max_epochs = 500
    npz = np.load('tutorial_data.npz')
    x_train = npz['train']
    x_val = npz['validate']
    #print(x_train.shape)

    #partition = {}
    training_set = Dataset(x_train, params['batch_size'])
    validation_set = Dataset(x_val, params['batch_size'])
    print('Got the files')
    # Generators
    #training_set = Dataset(partition['train'])
    training_generator = vxm_data_generator(x_train,batch_size=1)#training_set
    print('Made the training generator')

    #validation_set = Dataset(partition['validation'])
    validation_generator = vxm_data_generator(x_val,batch_size=1)#validation_set
    print('Made the validation generator')

    # Loop over epochs
    for epoch in range(max_epochs):
        start_time = time.time()
        train_loss = 0
        train_dice_score = 0
        val_loss = 0
        val_dice_score = 0
        count =0
        #for batch_fixed, batch_moving in training_generator:
        while count<100:
            batch_fixed, batch_moving = next(training_generator)
            #print(batch_fixed.shape, batch_moving.shape)
            batch_fixed = batch_fixed.float().cuda()
            batch_moving = batch_moving.float().cuda()
            loss, dice = vm.train_model(batch_moving, batch_fixed)
            train_dice_score += dice.data
            train_loss += loss.data
            count = count +1
        print('[', "{0:.2f}".format((time.time() - start_time) / 60), 'mins]', 'After', epoch + 1, 'epochs, the Average training loss is ', train_loss *
              params['batch_size'] / count, 'and average DICE score is', train_dice_score.data * params['batch_size'] / count)
        # Testing time
        start_time = time.time()
        vcount =0
        #if epoch % 1 == 0:
          #torch.save({
          #            'model_state_dict': vm
          #},'/result/model_unet.pt')
        while vcount<20:
            batch_fixed, batch_moving = next(validation_generator)
            # Transfer to GPU
            batch_fixed = batch_fixed.float().cuda()
            batch_moving = batch_moving.float().cuda()
            loss, dice = vm.get_test_loss(batch_moving, batch_fixed)
            val_dice_score += dice.data
            val_loss += loss.data
            vcount = vcount +1
        print('[', "{0:.2f}".format((time.time() - start_time) / 60), 'mins]', 'After', epoch + 1, 'epochs, the Average validations loss is ', val_loss *
              params['batch_size'] /vcount, 'and average DICE score is', val_dice_score.data * params['batch_size'] / vcount)

    return batch_fixed,batch_moving, vm


if __name__ == "__main__":
    main()
