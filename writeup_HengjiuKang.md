#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

## Introduction
This is homework at Self-driving car nanodegree from UDacity. This homework is done by Hengjiu Kang.

*Last three sections used some codes from [NikolasEnt](https://github.com/NikolasEnt/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb) to show the images*

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

![post_process](./images/post_process.png) "Random Noise"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

#Data Set Summary & Exploration

## Input data
Input data are pictures(in pixels) in 32x32x3 dimensions. The original data has RGB channels, but in my training process, I just use grascale pictures.

```python
training_file = "train.p"
validation_file= "valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
``` 
and the basic information is:
```
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43
```

## Pre-process
In this section, I built a 4-stage pipline processing the images. They are:
* Grayscale: 'shrink' RGB channels to one, simplifier the training data, and give it a better focus.
* EqualizeHist: Adjust the bright and dark region, make the picture more readable.
* Normalize: Conver the uint8 type to float point, make it possible to train the kernel.
* Gaussian noise: It is somehow very useful. After several tries, I found that adding gaussian noise is a good way to avoid overfit.

As the code below:
```python
def pre_process(image_data):
    result_set = []
    for i in range(len(image_data)):
        image = cv2.cvtColor(image_data[i], cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = noisy(image)
        result_set.append(np.expand_dims(image, axis=2))
    return result_set

```


#Design and Test a Model Architecture

At Step 3, I construct the net work, build the model and do the actual training work.
I modified the network based on LeNet-5. LeNet-5 is good at previous lab in recognizing the handwritten digits, but in traffic sign recognition, it only has about 5% correctness, which is considered as random choices.


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32     									|
| Sigmoid		|         									|
| Fully connected       | 800 input, 240 output                                         |
| Sigmoid				|        									|
|	Fully Connected				|	240 input, 43 output					|
|						|												| 

From step 3.2 to step 3.4, I extracted labels, setup training pipline and model evaluation process.


# Training process
At Step 3.5 I trained the model.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 95.7%
* validation set accuracy of 93.2% 
* test set accuracy of 83.33% (6 pictures)

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 