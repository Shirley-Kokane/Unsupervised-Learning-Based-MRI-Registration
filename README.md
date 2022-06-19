# Unsupervised-Learning-Based-MRI-Registration

In deformable registration, a dense, non-linear correspondence is established between a
pair of n-D image volumes, such as 3-D MR brain scans, depicting similar structures. 

Most registration methods solve an optimization problem for each volume pair that aligns
voxels with a similar appearance while enforcing smoothness constraints on the registration mapping. Solving this optimization is computationally intensive, and therefore extremely slow in practice.


We propose a novel registration method that learns a parameterized registration function from a collection of volumes. We implement the function using a convolutional neural network (CNN), that takes two n-D input volumes and outputs a mapping of all voxels of one volume to another volume.

![alt text](https://github.com/Shirley-Kokane/Unsupervised-Learning-Based-MRI-Registration/blob/main/register.png)
