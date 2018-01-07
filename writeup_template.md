# **Traffic Sign Recognition** 

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: https://github.com/deepanshu96/carp2/blob/master/ger/ger1.jpg
[image5]: https://github.com/deepanshu96/carp2/blob/master/ger/ger1.jpg
[image6]: https://github.com/deepanshu96/carp2/blob/master/ger/ger1.jpg
[image7]: https://github.com/deepanshu96/carp2/blob/master/ger/ger1.jpg
[image8]: https://github.com/deepanshu96/carp2/blob/master/ger/ger1.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar graph showing the number of images per class in each of the training, validation and test set.

![alt text](https://github.com/deepanshu96/carp2/blob/master/Imag/Screen%20Shot%202018-01-06%20at%208.26.07%20PM.png)

![alt text](https://github.com/deepanshu96/carp2/blob/master/Imag/Screen%20Shot%202018-01-06%20at%208.26.14%20PM.png)

![alt text](https://github.com/deepanshu96/carp2/blob/master/Imag/Screen%20Shot%202018-01-06%20at%208.26.23%20PM.png)

I also represented each class with an example image with it.

![alt text](https://github.com/deepanshu96/carp2/blob/master/Imag/Screen%20Shot%202018-01-06%20at%208.26.41%20PM.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

*In the first step I converted the given training,test and validation data set into grayscale.

![alt text](https://github.com/deepanshu96/carp2/blob/master/Imag/Screen%20Shot%202018-01-06%20at%208.27.02%20PM.png)

*In the next step I normalized the given training,test and validation data set.

![alt text](https://github.com/deepanshu96/carp2/blob/master/dib.png)

*I also generated the additional data for training and valdidation sets but in the end I did not use it because the validation accuracy due to the additional data was coming out to be very low and not up to the mark according to the rubric points. The transformed additional data is shown below :-

![alt text](https://github.com/deepanshu96/carp2/blob/master/Imag/Screen%20Shot%202018-01-06%20at%208.27.11%20PM.png)

*The transformed data that is training and validation data sets distribution is shown below :-

![alt text](https://github.com/deepanshu96/carp2/blob/master/Imag/Screen%20Shot%202018-01-06%20at%208.27.24%20PM.png)

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 1x1x400 	|
| RELU					|												|
| Flatten Layers		| 5x5x16 -> 400,  1x1x400->400     		|
| Concatenate Flatten Layers | Input = 400,400 Output = 800             |
| Dropout |   |
| Fully Connected Layers		| Input = 800. Output = 120     		|
| Fully Connected Layers		| Input = 120. Output = 84     		|
| Fully Connected Layers		| Input = 84. Output = 43     		|
| Softmax	Layer			|         									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with learning rate = 0.001, batch size = 128 and number of epocs = 60.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.941 
* test set accuracy of 0.926

I used a model similar to the Sermanet and LeCun model to classify the traffic sign images. At first I preprocessed the images which included transformed images also but the validation accuracy was quite low,then I used only the normalized images to obtain the above given validation accuracy. Also I adjusted the batch size, dropout probability and number of epochs to obtain the results.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, which have been turned to grayscale and have been normalized :-

![alt text](https://github.com/deepanshu96/carp2/blob/master/dola.png)

The pictures might be difficult to classify because they have been reduced from different sizes available on the internet to 32X32 images. This can result in significant data losses as some pictures had different aspect ratio.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model gave an accuracy of 0.50 when run on the above 6 images. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top 5 softmax probabilities along with the top 5 class for each image as predicted by the model are given below :-

![alt text](https://github.com/deepanshu96/carp2/blob/master/dola2.png)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


