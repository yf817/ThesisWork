# ThesisWork
# Unsupervised Global To Local Nonrigid Image Registration Network Bidirectionally Reinforced By Joint Saliceny Map

## Abstract
- After decades of scientific development, non-rigid image registration is becoming an indispensible image processing tool for many research areas such as life science, medical imaging, motion analysis and pattern recognition. On the one hand, with rapid development of imaging technologies and upgrading devices, the surging of large scale, high-dimensional images with rich structural and functional information has to cope with the large computational cost of image registration. On the other hand, due to common reasons like biological tissue motion, tumor development and outside physical factors, the images to be registered often contain outliers such as missing correspondences and large local deformations, which greatly challenge deformable image registration in making the balance between accuracy and robustness. 

- To address the missing correspondences and large local deformations in an accurate, robust and fast manner, we propose an unsupervised deformable image registration deep neural network that firstly predict a global deformation field and then elevate the accuracy of global prediction with local regression. Registration deep network aims to learn the pixelwise mapping between moving image and target image while this mapping function is too complicated by the outliers to be learned in the deep neural network. In our work, we take a "divide and conquer" tactics that decompose the original complex mapping into two solvable sub-mappings: a first global prediction network learns the mapping form input images to coarse motion field, and a subsequent local regression network is to accurately recover true motion field from coarse deformation prediction.

- To curb the vicious deformations at the outlier regions, we further introduce a joint saliency map based bidirectional reinforcement mechanism into the registration network. Taking full advantage of consistency relationship between images’ structural context and deformation fields indicated by joint saliency map, the proposed network focuses more on correctly-aligned joint salient pixels and suppresses the vicious impact from erroneously registered pixels and outlier pixels with missing correspondences. Differentiable joint saliency map not only bidirectionally reinforces the whole network’s cooperative training by providing image contextual guidance for forward prediction of local regression network and generating backward feedback gradients basing on image alignment residual error to improve the global prediction network’s learning, but also prompt fast convergence to accurate and robust deformable registration results at both global and local estimation. 

- Experiment results demonstrate that the proposed unsupervised global to local deformable registration network bidirectionally reinforced by joint saliency map is able to tackle with difficult deformable registration problem with missing correspondences and large local deformations in an accurate, robust and fast manner. Compared with several state-of-art traditional and deep learning based registration methods, our method achieved best in both visual evaluation and landmark-based quantitative evaluation and the running time is reduced to 0.01~0.2 second in 2D image registration.    

- KEY WORDS: Non-Rigid Image Registration, Joint Saliency Map, Deep Learning, Deep Neural Network, Bidirectional reinforcement, Convolutional Neural Network, Global to Local

## Proposed Model
![Proposed Model](https://github.com/fedral/ThesisWork/raw/model.jpg)

1. [x] create_HD5dataset
- 1.1 creating H5DF Dataset  
2. [x] flownet_supervised
- 2.1 helper
  - Loss function
  - Batch generator using dask multi-threads 
  - Flow R/W libs
- 2.2 layer
  - warpinglayer based on Spatial Transformer network 
3. [x] flownet_unsupervised

- Envirnment: GTX1080Ti,I7; Lasagne 02.dev/Theano 0.90; Ubuntu 14.04; Cudnn 5.0 + Cuda 8.0
- Testing   : Lasagne 02.dev/Theano 0.90; Tensorflow 0.12/Keras; Caffe;
## Tesing Result
![Comparison Result with satus quo both traditional and deep learning based image registration algorithms](https://github.com/fedral/ThesisWork/raw/errorplot.jpg)


## Reference
1. Jaderberg, Max, Karen Simonyan, and Andrew Zisserman. "Spatial transformer networks." Advances in Neural Information Processing Systems. 2015.
2. Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.
3. Dosovitskiy, Alexey, et al. "Flownet: Learning optical flow with convolutional networks." Proceedings of the IEEE International Conference on Computer Vision. 2015.
4. Ilg, Eddy, et al. "Flownet 2.0: Evolution of optical flow estimation with deep networks." arXiv preprint arXiv:1612.01925 (2016).
5. Qin, Binjie, Shen Z, Fu Z,et al. Joint-saliency structure adaptive kernel regression with adaptive-scale kernels for deformable registration of challenging images[J]. IEEE Acce. 2018, 6: 330-343.

