# Part-III-project
# Guidelines

## 1. Problem Statement
My project is called "Deep Learning to track fringes". In interferometric observations of faint astronomical objects, fringe patterns are often mixed with background noise. When the signal-to-noise (SNR) ratio is decent, human eyes are quite good at distinguishing the fringe from the noise, however, when the SNR drops, it becomes increasing harder. The aim of this project is to exploit deep learning methods to train a machine that can overcome this difficulty.

## 2. Packages
Firstly import all the packages that you will need for this project.
- [numpy](www.numpy.org) is the fundamental package for scientific computing with Python.
- [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.
- [h5py](http://www.h5py.org) is a common package to interact with a dataset that is stored on an H5 file.
- [PIL](http://www.pythonware.com/products/pil/) and [scipy](https://www.scipy.org/) are used here to test your model with your own picture at the end.
- dnn_app_utils provides the functions needed to build a L-layer NN.
- np.random.seed(1) is used to keep all the random function calls consistent.

## 3. Dataset
You need to partition your data into training, dev and test sets.
- Training set is used to train the neural network parameters for different models.
- Dev set is used to compare the cost of those different models.
- Test set is used to test the accuracy of the trained model.

Depending on the size of your data, you may need to do **pre-training** on other available data set and use your own data to **fine-tune** on the trained model.

### 3.1 Example Data
The data in this project are generated using David Buscher's Python codes. Fringe pattern under different light level looks like this:

<img src="/Users/leo/Documents/GitHub/Part-III-Project/NN_python_codes/group-delay-track-high-light.jpg" width="300">
High light level with SNR=10.0

<img src="/Users/leo/Documents/GitHub/Part-III-Project/NN_python_codes/group-delay-track-low-light.jpg" width="300">
Low light level with SNR=1.0

### 3.2 Analysis of Data
The fringe pattern is generated using a txt file containing around 130,000 phase sequence values. Each time a section of the phase sequence is chosen and used in the PlotGroupDelaySimulation function.  

The background noise is generated using the random normal distribution function in the numpy package. The different level of background noise is controlled by the SNR factor. The fact that human eyes can barely identify the fringe once the SNR drops below 1 might serve as a good estimate of the Bayesian error. (**Need revise**)

Then fringe and noise are combined together to arrive at the final image. Concretely, the image has *time* as *x-axis* and *delay wavelength* as *y-axis*.

How do we get enough training and test examples? ~~I propose the following parallel routes:~~
- ~~We can partition the 130,000 phase sequence values into say, more than 10,000 intersecting continuous set, and use those set with a fixed SNR to get the images. **But is this plausible?**~~
- ~~We can use pre-trained de-noising model on the limited number of our training set and fine-tune the model to fit the requirement. How do we get the pre-trained model? Maybe try searching on the web for similar fringe pattern with noise?~~
- use ```cnn_generate_image.py``` to generate as many training and test images as you want.



## 4. Architecture of the Model
Essentially the project boils down to image de-noising, which is an unsupervised ML problem. Based on the structure of a neural network, we can use the auto-encoder/decoder combined with sparsity coding to tackle this problem. Notes on [sparse auto-encoder](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial) can be found online.

### 4.1 Problem Formulation
Assuming $\textbf{x}$ is the observed noisy image and $\textbf{y}$ is the original noise free image, we can formulate the image corruption process as:
\[
\textbf{x} = \eta(\textbf{y})
\]
where $\eta: \Re^{n} \rightarrow \Re^{n}$ is an arbitrary stochastic corrupting process that input noise into our image, in this case, it is a random normal distribution. Then the de-noising task's learning objective is to find a function $f$ that best approximates $\eta^{-1}$.

### 4.2 Auto-encoder
Suppose we have only a set of unlabelled training examples {${x^{(1)}, x^{(2)}, x^{(3)}, \dots}$}, where $x^{(i)}$ $\in$ $\Re^{n}$. An **autoencoder** neural network is an unsupervised learning algorithm that applies back-propagation, setting the target values to be equal to the inputs. i.e., it uses $y^{(i)} = x^{(i)}$. An example of auto-encoder looks like this:

<img src="/Users/leo/Documents/GitHub/Part-III-Project/NN_python_codes/Autoencoder.png" width="300">

The auto-encoder tries to learn a function $h_{W,b}(x)\approx x$. In other words, it is trying to learn an approximation to the identity function, so as to output $\hat{x}$ that is similar to $x$. The identity function seems a particularly trivial function to be trying to learn; but by placing constraints on the network, such as by limiting the number of hidden units, we can discover interesting structure about the data. As a concrete example, suppose the inputs $x$ are the pixel intensity values from a 10 × 10 image (100 pixels) so $n = 100$, and there are $s_2 = 50$ hidden units in layer $L_2$. Note that we also have $y \in \Re^{100}$. Since there are only 50 hidden units, the network is forced to learn a compressed representation of the input, i.e., given only the vector of hidden unit activations $a^{(2)} \in \Re^{50}$, it must try to **reconstruct** the 100-pixel input $x$. If the input were completely random—say, each $x_i$ comes from a 2D Gaussian independent of the other features—then this compression task would be very difficult. But if there is structure in the data, for example, if some of the input features are correlated, then this algorithm will be able to discover some of those correlations.

For our problem, we will twist a bit on the definition of the auto-encoder. Let $\textbf{y}_{i}$ be the original data for $i=1,2,\dots,N$ and $\textbf{x}_{i}$ be the corrupted version of corresponding $\textbf{y}_i$, the auto-encoder will try to learn a function that given any $x$, it will return a $\hat y$ that best approximates $y$.

### 4.3 Sparsity
When the number of hidden units is large (perhaps even greater than the number of input pixels), we can still discover interesting structure, by imposing a sparsity constraint on the hidden units, then the auto-encoder will still discover interesting structure in the data. Sparsity is achieved by constraining the neurons to be inactive most of the time. 'Inactive' here means an output value of 0 from a sigmoid activation function, or -1 from a $\tanh$ activation function. In particular, if we write $a_j^{(2)}(x)$ as the activation of the hidden unit $j$ in the second layer when fed by a given input $x$, the average activation of hidden unit $j$ (averaged over the training set) is defined by:
\[
\hat{\rho}_j=\dfrac{1}{m}\sum_{i=1}^{m}[a_j^{(2)}(x^{(i)})]
\]
We would like the following constraint to be satisfied:
\[
\hat{\rho}_j=\rho
\]
where $\rho$ is the sparsity parameter, typically set to a small value close to zero (e.g. $\rho = 0.05$). To achieve this, a penalty term must be added to the cost function $J$ that penalise $\hat{\rho}_j$ deviating significantly from $\rho$. As an example, we can use the Kullback-Leiber (KL) divergence to define the penalty term:
\[
\sum_{j=1}^{s_2}\text{KL}(\rho||\hat{\rho}_j)
\]
where $\text{KL}(\rho||\hat{\rho}_j)=\rho \log\dfrac{\rho}{\hat{\rho}_j}+(1-\rho)\log\dfrac{1-\rho}{1-\hat{\rho}_j}$ measures how different two distributions are, and $s_2$ is the number of neurons in the hidden layer.  The penalty term has the property that $\text{KL}(\rho||\hat{\rho}_j)=0$ if $\hat{\rho}_j=\rho$, and increases as $\hat{\rho}_j$ diverges from $\rho$. The overall cost function can be written as:
\[
J_{sparse}(W,b) = J(W,b) + \beta\sum_{j=1}^{s_2}\text{KL}(\rho||\hat{\rho}_j)
\]
where $J(W,b)$ is the ordinary cost function associated with a neural network, and $\beta$ is the coefficient that controls the weight of the sparsity penalty term.

### 4.4 Implementation Notes on Sparsity Term
To implement the KL divergence term into your derivative calculation, simply add the KL gradient term:
\[
\beta(-\dfrac{\rho}{\hat{\rho}_j}+\dfrac{1-\rho}{1-\hat{\rho}_j})
\]
You need to compute a forward propagation on all the training examples to compute the average activations on the training set, before computing backward_propagation on any example.

## 5. Alternative Architecture
I had been thinking about using sparsity to reduce overfitting, however, it came to my mind that I can exploit existing architectures like CNNs to train my data.

### 5.1 Dropout technique
Dropout is widely used regularisation technique that is specific to deep learning. It randomly shuts down some neurons in each iteration. At each iteration, you shut down (i.e. set to zero) each neuron of a layer with probability (1-*keep_prob*) or keep it with probability *keep_prob*. The dropped neurons don't contribute to the training in both the forward and backward propagation of the iteration.

The idea behind dropout is that at each iteration, you train a different model that uses only a subset of your neurons. With dropout, your neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time.

### 5.2 Residual Learning of Deep Convolutional Neural Network
Convolutional neural networks is a class of deep, feed-forward artificial neural networks that has successfully been applied to analyzing visual imagery. CNNs use relatively little pre-processing compared to other image classification algorithms. This means that the network learns the filters that in traditional algorithms were hand-engineered. This independence from prior knowledge and human effort in feature design is a major advantage.

The reasons of using CNN are three-fold. First, CNN with very deep architecture is effective in increasing the capacity and flexibility for exploiting image characteristics. Second, considerable advances have been achieved on regularization and learning methods for training CNN, including Rectifier Linear Unit (ReLU), batch normalization and residual learning. These methods can be adopted in CNN to speed up the training process and improve the denoising performance. Third, CNN is well-suited for parallel computation on modern powerful GPU, which can be exploited to improve the run time performance.


## 6. Problems/Ideas
My data can be read/interpreted in terms of the format they are presented. It can either be fed as a pure image with grey scale pixel values at different places, or it can be put as just x/y values from its generating data. Which format should I use?

- If I use the image format, it has the advantage of not having to think about individual data point, yet it needs reshape of its pixels to a single vector. And I don't know a priori what the dimension of my image is.
- It's vice versa for the other input format.


### 6.1 Image Input Format

#### 6.1.1 Training set
When we talk about deep learning, usually the first thing comes to mind is a huge amount of data or a large number of images (e.g. a couple of millions images in ImageNet). In such situation, it is not very smart and efficient to load every single image from the hard drive separately and apply image preprocessing and then pass it to the network to train, validate, or test. Despite the required time to apply the preprocessing, it's way more time consuming to read multiple images from a hard drive than having them all in a single file and read them as a single bunch of data. Hopefully, there are different data models and libraries which come out in favour of us, such as HDF5 and TFRecord. In this post we learn how to save a large number of images in a single HDF5 file and then load them from the file in batch-wise manner. It does not matter how big the data is and either it is larger than your memory size or not. HDF5 provides tools to manage, manipulate, view, compress and save the data. [Link.](http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html)


#### 6.1.2 Testing the trained model
For test images, I need to resize them and trim them using ```resize.py``` and ```trim.py```. The test image (with its corresponding clean image) can then  be fed into the learning neural network.



### 6.2 Data Generation
The fringe pattern is generated using the phase sequences from ```temprl2.py```. The image is generated using ```cnn_generate_image.py``` and saved into corresponding files. This method can be used to generate both the training and testing sets as each call of ```temprl2.py``` outputs randomly seeded phases.

### 6.3 Measurement of Image Quality
Comparing restoration results requires a measure of image quality. Two commonly used measures are Mean-Squared Error and Peak Signal-to-Noise Ratio [Ref](Rafael C. Gonzalez and Richard E. Woods. Digital Image Processing. Addison-Wesley, New York, 1992.
). The mean-squared error (MSE) between two images $g(x,y)$ and $\hat{g}(x,y)$ is:
\[
\text{MSE} = \dfrac{1}{MN}\sum_{n=1}^{N}\sum_{m=1}^{M}[\hat{g}(n,m)-g(n,m)]^2
\]
One problem with mean-squared error is that it depends strongly on the image intensity scaling. A mean-squared error of 100.0 for an 8-bit image (with pixel values in the range 0-255) looks dreadful; but a MSE of 100.0 for a 10-bit image (pixel values in [0,1023]) is barely noticeable.

Peak Signal-to-Noise Ratio (PSNR) avoids this problem by scaling the MSE according to the image range:
\[
\text{PSNR} = -10log_{10}\dfrac{\text{MSE}}{S^2}
\]
where $S$ is the maximum pixel value. PSNR is measured in decibels (dB). The PSNR measure is also not ideal, but is in common use. Its main failing is that the signal strength is estimated as   , rather than the actual signal strength for the image. PSNR is a good measure for comparing restoration results for the same image, but between-image comparisons of PSNR are meaningless. One image with 20 dB PSNR may look much better than another image with 30 dB PSNR.




## 7. Implementation Notes
To provide data suitable for training, I need to remove the x/y labels and axises. This is done by adding a few extra codes in the file ```GroupDelaySimulation_leo.py```which generates plots.

## 8. Comparison
As a comparison with other state-of-art de-noising techniques, I tried the OpenCV method using the Non-local Means De-noising algorithm. For a noisy image I get the following result:
<img src="/Users/leo/Documents/GitHub/Part-III-Project/NN_python_codes/cv2denoising.png" width="300"> De-noised version

<img src="/Users/leo/Documents/GitHub/Part-III-Project/NN_python_codes/group-delay-track-low-light.jpg" width="300"> Noisy version

We notice that this technique does not really achieve a great result - the noise does not really get removed to a satisfying degree.

For our DnCNN architecture (L6F64), I get
<img src="/Users/leo/Documents/GitHub/DnCNN-tensorflow-master/sample/L6F64/test1_2400.png" width="500">
(Clean/Noisy/Denoised)


## 9. Literature Review
The state-of-art algorithms for image de-noising using deep learning are as follows:
- [TNRD](https://arxiv.org/pdf/1508.02848.pdf)
  - Trainable nonlinear reaction diffusion: A flexible framework for fast and effective image restoration (Chen et al. 2017).
- [DnCNN](https://arxiv.org/pdf/1608.03981v1.pdf)
  - Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (Zhang et al. 2017).
- [DAAM](https://arxiv.org/pdf/1612.06508.pdf)
  - Deeply Aggregated Alternating Minimisation for Image Restoration (Youngjung Kim et al. 2016).
- [Adversirial Denoising](https://arxiv.org/pdf/1708.00159.pdf)
  - Image Denoising via CNNs: An Adversirial Approach (Divakar et al. 2017)
- [Unrolled Optimisation Deep Priors](https://arxiv.org/pdf/1705.08041.pdf)
  - Unrolled Optimisation with Deep Priors (Diamond et al. 2017)
- [Wider Network](https://arxiv.org/pdf/1707.05414.pdf)
  - Going Wider with Convolution for Image Denoising (Liu et al. 2017)
- [Recurrent Inference Machines](https://arxiv.org/pdf/1706.04008.pdf)
  - Recurrent Inference Machines for Solving Inverse Problems (Putzky et al.)
- [Learning Pixel Distribution Prior](https://arxiv.org/pdf/1707.09135.pdf)
  - Learning Pixel Distribution Prior with Wider Convolution for Image Denoising (Liu et al. 2017)


## 10. To-Do
- [x] Check [this.](https://github.com/crisb-DUT/DnCNN-tensorflow) Managed to use this CNN model on my local laptop. Should move on to train my own data to see the effect.
  - To train this model, follow these steps in the DnCNN-tensorflow-master folder:
  ```shell
   python3 main.py --phase train --epoch 20
  ```
  - To test this model, run:
  ```shell
   python3 main.py --phase test
  ```

- [x] How to crop my images to reduce the picture size?   
  ```shell
  trim.py
  ```
- [x] Literature Review
- [x] Analyse Data Structure
- [x] Generate dataset
  - [x] Training and Test sets
- [x] Modify the codes of my neural networks
  - [x] Modify the last layer to output clean data
  - [x] Modify the cost functions
- [x] Train my model and test it
  - [x] If successful, consider implementing more advanced architectures mentioned in the literature review
- [x] Compare different algorithms
  - [x] Compare models with different layers and sizes.
  - [x] Compare models with different architectures.
- [x] Write the presentation
- [x] Write the report

## 11. Progress Check

- Done general study on machine learning and deep neural network (took ~7 weeks).
- Implemented a simple multi-layer neural network from scratch, but it turned out to be too slow on my dataset.
- Adopted current state-of-art deep learning architecture (Residual CNN) using Tensorflow to train my dataset, speed increased significantly on test samples. The result is not satisfying due to the fact that I only used a training sample of 40 images. My laptop can't handle larger set, so I will consider using department machines to do the full training.
- ~~Data generation is still a problem, discuss with David. Is augmentation a good idea in our case? But if we use augmentation, how should we keep track of the intrinsic noisy images?~~
  - ~~Currently have two different approaches of generating noisy images:~~
    - ~~Add noise before processing, which is in David's codes already, each clean image has one corresponding noisy image. But this way is difficult to generate more data.~~
    - ~~Add artificial noise after augmentation, easier for generating more data, but not intrinsic any more. (Maybe consider adding blind Gaussian noise?)~~


## 12. Training

### 12.0 Preparation
- Wrote script to generate 80 clean images with 80 corresponding noisy images from phase sequence data.
- Wrote script to trim the images by concentrating only on the fringe (the resulting images might have different size) - the purpose of this is to reduce the amount of data our network will handle.
- We can generate as many different images as we want since the phase sequences are randomly generated each time we call it.



### 12.1 Changing the number of layers L
As shown in Fig. 2(a), CNN model with the number of layers L = 5 has the remarkable increase in performance than the network with L = 3. As L in each layer increases, the performance improves only slightly, but with the corresponding training time and computation complexity growing vastly.


### 12.2 Changing the number of filters F



### 12.3 Changing the size of filters K


### 12.4 Sum up
To sum up, from the overall performance and efficiency point of view, CNN with L = 5, K = 128, F = 7 x 7 is potentially the optimal model among plain shallow CNNs.

### 12.5 Large dataset
Now with the optimal model we can try to train it on a large dataset without the images being trimmed.

## 13. Project Write-up
### 13.1 Abstract
### 13.2 Introduction
#### 13.2.1 Physic background
#### 13.2.2 Deep learning background
### 13.3 Method
#### 13.3.1 Simulating fringe motions
#### 13.3.2 Deep learning architecture
### 13.4 Results
#### 13.4.1 Training
#### 13.4.2 Testing
### 13.5 Discussion
### 13.6 Conclusion
