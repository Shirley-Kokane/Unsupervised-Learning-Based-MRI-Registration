# Unsupervised-Learning-Based-MRI-Registration

In deformable registration, a dense, non-linear correspondence is established between a
pair of n-D image volumes, such as 3-D MR brain scans, depicting similar structures. 

Most registration methods solve an optimization problem for each volume pair that aligns
voxels with a similar appearance while enforcing smoothness constraints on the registration mapping. Solving this optimization is computationally intensive, and therefore extremely slow in practice.


We propose a novel registration method that learns a parameterized registration function from a collection of volumes. We implement the function using a convolutional neural network (CNN), that takes two n-D input volumes and outputs a mapping of all voxels of one volume to another volume.

![alt text](https://github.com/Shirley-Kokane/Unsupervised-Learning-Based-MRI-Registration/blob/main/register.png)


There are two types of algorithms in image registration. First is Rigid Image Registration(RIR) and the second is Deformation Image Registration (DIR). The process in which all transformations are affine that is the pixel to pixel relationship remains the same as before is known as RIR. This is a linear method and frequently used in the past. It is useful when the moving image has no deformity. The major drawback of this method is that it cannot be used when the moving image incurred some deformation. This happens quite often in medical images when there is a disease like a tumor which can grow or shrink with time. Deformation image registration(DIR) process is used in such cases.

DIR methods are employed when RIR cannot perform the desired task. They can be used to analysis and comparison of medical structures between the scans. Such analysis is used to assess and understand the evolution of brain anatomy over time for individuals with the disease. Deformable registration strategies often involve two steps: an initial affine transformation for global alignment, followed by a much slower deformable transformation with more degrees of freedom. We concentrate on the latter step, in which we compute a dense, nonlinear correspondence for all pixels.

![alt text][https://github.com/Shirley-Kokane/Unsupervised-Learning-Based-MRI-Registration/blob/main/download.png]

## How to use

```python

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
        for batch_fixed, batch_moving in training_generator:
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

```
