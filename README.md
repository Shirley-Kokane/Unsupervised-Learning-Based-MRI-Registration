# Unsupervised-Learning-Based-MRI-Registration

In deformable registration, a dense, non-linear correspondence is established between a
pair of n-D image volumes, such as 3-D MR brain scans, depicting similar structures. 

Most registration methods solve an optimization problem for each volume pair that aligns
voxels with a similar appearance while enforcing smoothness constraints on the registration mapping. Solving this optimization is computationally intensive, and therefore extremely slow in practice.


We propose a novel registration method that learns a parameterized registration function from a collection of volumes. We implement the function using a convolutional neural network (CNN), that takes two n-D input volumes and outputs a mapping of all voxels of one volume to another volume.

![alt text](https://github.com/Shirley-Kokane/Unsupervised-Learning-Based-MRI-Registration/blob/main/register.png)


There are two types of algorithms in image registration. First is Rigid Image Registration(RIR) and the second is Deformation Image Registration (DIR). The process in which all transformations are affine that is the pixel to pixel relationship remains the same as before is known as RIR. This is a linear method and frequently used in the past. It is useful when the moving image has no deformity. The major drawback of this method is that it cannot be used when the moving image incurred some deformation. This happens quite often in medical images when there is a disease like a tumor which can grow or shrink with time. Deformation image registration(DIR) process is used in such cases.

DIR methods are employed when RIR cannot perform the desired task. They can be used to analysis and comparison of medical structures between the scans. Such analysis is used to assess and understand the evolution of brain anatomy over time for individuals with the disease. Deformable registration strategies often involve two steps: an initial affine transformation for global alignment, followed by a much slower deformable transformation with more degrees of freedom. We concentrate on the latter step, in which we compute a dense, nonlinear correspondence for all pixels.
