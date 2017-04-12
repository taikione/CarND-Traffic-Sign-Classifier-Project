# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/1_speed_limit_30.jpeg "sign1"
[image2]: ./examples/17_no_entry.jpeg "sign2"
[image3]: ./examples/18_general_coution.jpeg "sing3"
[image4]: ./examples/21_double_curve.jpeg "sign4"
[image5]: ./examples/38_keep_right.jpeg "sign5"

[test_detail]: ./images/Test_dataset_details.png "test_detail"
[train_detail]: ./images/Train_dataset_details.png "train_detail"
[valid_detail]: ./images/Validation_dataset_details.png "valid_detail"
[test_imgs]: ./images/test_data.png "test_imgs"
[train_imgs]: ./images/training_data.png "train_imgs"
[valid_imgs]: ./images/validation_data.png "valid_imgs"
[grayscale]: ./images/grayscale.png "grayimgs"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 $\times$ 32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the dataset. Following graph and figure show details of training dataset and each class examples.

![traindata_details][train_detail]
![train_imgs][train_imgs]

Validation dataset and Test dataset are also shown.

![valid_details][valid_detail]
![valid_imgs][valid_imgs]
![test_details][test_detail]
![test_imgs][test_imgs]

I confirmed that each dataset has a similar configuration and bias.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to convert the images to grayscale because I create small and simple model. I consider that simple model is good way to prevent overfitting.

Here is an example of a traffic sign image before and after grayscaling.

![preprocessing][grayscale]
Preprocessing I used is grayscale only. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScale image   		    		|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 		    		|
| RELU					| activate function								|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| Max pooling  			| 2x2 stride, outputs 5x5x16 					|
| RELU					| activate function								|
| Fully connected		| outputs 120       							|
| Dropout				| keep_prob 0.5	(only tested)					|
| RELU					| activate function								|
| Fully connected		| outputs 84        							|
| Dropout				| keep_prob 0.5									|
| RELU					| activate function								|
| Output				| outputs 43									|
| Softmax				|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following parameters:
- Optimizer : Adam Algorithm
- Epochs : 30
- Batch size : 128
- Learning rate : 0.001
- Dropout rate : 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.990. In[13] Cell
* validation set accuracy of 0.932. In[12] Cell
* test set accuracy of 0.921. In[22] Cell

My first model is same as LaNet architecture. But, I did not get accuracy as I expected. In case of using first model, I may get over 0.93 accuracy considering increasing training Epochs. However, I wanted to get over 0.93 accuracy in 20 Epochs. Therefore, I added dropout to fully connected layers in second model. The reason of add to dropout is to prevent overfitting and improve accuracy. By adding dropout, I confirmed Network was able to learn properly even after 10 epoch. (I don't change optimizer, learning rate and batch size.) It was first improvement, but I was able to create a successful model. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] ![alt text][image4] ![alt text][image5]

The 4th image (double curve sign) might be a little difficult to classify because it was tilt.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 30 km/h   | Speed limit (30km/h)   						| 
| No entry     			| No entry 										|
| General coution		| General coution								|
| Double curve	      	| Dangerous curve to the right	 				|
| Keep right			| Keep right        							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This accuracy compares favorably with test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 28th cell of the Ipython notebook.

For the 1st image, the model is relatively sure that this is a speed limit (30km/h) sign (probability of 0.99), and the image does contain a speed limit (30km/h) sign. The top five softmax probabilities were following.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99999762e-01       	| Speed limit (30km/h)   						| 
| 1.83847831e-07     	| Speed limit (50km/h) 							|
| 1.96129388e-08		| Speed limit (80km/h)							|
| 2.26349733e-12	   	| Speed limit (70km/h) 				 			|
| 2.05891517e-15	    | End of speed limit (80km/h)	  				|

For the 2nd image, The top five softmax probabilities were following.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.56088305e-01       	| No entry				   						| 
| 4.39112000e-02     	| Stop				 							|
| 2.27572343e-07		| Priority road									|
| 1.64099561e-07	   	| Keep right		 				 			|
| 4.29053237e-08	    | Turn left ahead				  				|

For the 3rd image, The top five softmax probabilities were following.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00       	| General caution								| 
| 9.72636305e-10     	| Traffic signals	 							|
| 3.66364003e-13		| Pedestrians									|
| 1.52986113e-18	   	| Dangerous curve to the right		 			|
| 6.93137933e-21	    | Right-of-way at the next intersection			|

For the 4th image, the image does not contain a double curve sign(correct sign). The top five softmax probabilities were following.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 5.98050952e-01       	| Dangerous curve to the right  				| 
| 3.99354398e-01     	| Children crossing								|
| 1.06019538e-03		| Slippery road									|
| 9.42900660e-04	   	| Bicycles crossing		 			 			|
| 2.99373845e-04	    | Bumpy road					  				|

For the 5th image, The top five softmax probabilities were following.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00       	| Keep right					  				| 
| 7.40823115e-12     	| Priority road									|
| 1.87474268e-15		| End of all speed and passing limits			|
| 4.29362857e-19	   	| End of no passing		 			 			|
| 3.83374949e-19	    | Yield							  				|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


