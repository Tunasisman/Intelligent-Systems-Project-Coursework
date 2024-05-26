# Intelligent-Systems-Project-Coursework -- Flower Species Classification with Convolutional Neural Networks

**Project Overview**

This project aims to classify flower species using the Oxford 102 Flower dataset. A custom Convolutional Neural Network (CNN) was developed and trained from scratch using PyTorch. The network was designed to automatically learn features specific to the Flowers-102 dataset and achieved a test accuracy of 64.22%.

**Code Structure**

The project includes several scripts and files:

- train_model.py: Script to train the CNN model from scratch.
- evaluation.py: Script to evaluate the trained model and report validation and test accuracies.
- evaluation_and_testing.ipynb: Jupyter notebook for evaluating the model and visualizing predictions on random test images.

**Requirements**

Before running the code, ensure you have the following libraries installed:

- torch
- torchvision
- numpy
- matplotlib
  
You can install these libraries using pip:

" pip install torch torchvision numpy matplotlib "

**Training the Model**

To train the model from scratch, run the train_model.py script. This script will:

- Download the Oxford 102 Flower dataset.
- Preprocess the data using data augmentation techniques.
- Initialize and train the CNN model.
- Save the best-performing model.
  
*Running the Training Script*

" python train_model.py "

**Evaluating the Model**

To evaluate the trained model, use the evaluation.py script. This script will load the saved model and compute the validation and test accuracies.

*Running the Evaluation Script*
" python evaluation.py "

**Visualizing Predictions**

For a more interactive evaluation, use the evaluation_and_testing.ipynb notebook. This notebook will:

- Load the saved model.
- Evaluate the model on the validation and test sets.
- Randomly select 5 images from the test set and display the true class along with the model’s top 3 predictions.

*Running the Notebook*

- Open evaluation_and_testing.ipynb in Jupyter Notebook or JupyterLab.
- Run all cells to see the evaluation metrics and visualizations.
- To visualize predictions on different images, run the cell containing the test_and_visualize function multiple times.

**Hardware and Environment**

The experiments were conducted on a MacBook Pro M2 with a 19-core GPU. The training and evaluation processes were implemented using PyTorch. The code is compatible with both CUDA and MPS, allowing it to run on various GPU architectures. If you have a different GPU, ensure you modify the device configuration accordingly.

**Key Hyperparameters**
- *Optimizer:* Adam with a learning rate of 0.0001 and weight decay of 1e-5.
- *Learning Rate Scheduler:* Cosine Annealing with 𝑇max set to 100.
- *Batch Size:* 32
- *Epochs:* Up to 600 with early stopping if validation loss does not improve for 50 consecutive epochs.
- *Data Augmentation:* Random resized cropping, horizontal flipping, random rotation, color jittering, and normalization.

**Early Stopping**
The training process includes an early stopping mechanism to prevent overfitting. If the validation loss does not improve for 70 consecutive epochs, training is halted.



**References**
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. Advances in Neural Information Processing Systems, 25.
- Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- Nilsback, M.-E., & Zisserman, A. (2008). Automated Flower Classification over a Large Number of Classes. Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems, 32.
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning. Journal of Big Data, 6(1), 60.
- Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic Gradient Descent with Warm Restarts. arXiv preprint arXiv:1608.03983.

