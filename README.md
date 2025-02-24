Download link :https://programming.engineering/product/ece421-assignment-1-logistic-regression/


# ECE421-Assignment-1-Logistic-Regression
ECE421 – Assignment 1: Logistic Regression
Objectives:

In this assignment, you will rst implement a simple logistic regression classi er using Numpy and train your model by applying (Stochastic) Gradient Descent algorithm. Next, you will implement the same model, this time in TensorFlow and use Stochastic Gradient Descent and ADAM to train your model.

You are encouraged to look up TensorFlow APIs for useful utility functions, at: https://www.

tensorflow.org/api_docs/python/.

General Note:

Full points are given for complete solutions, including justifying the choices or assumptions you made to solve each question.

A written report should be included in the nal submission. Do not dump your codes and outputs in the report. Keep it short, readable, and well-organized.

Programming assignments are to be solved and submitted individually. You are encouraged to discuss the assignment with other students, but you must solve it on your own.

Please ask all questions related to this assignment on Piazza, using the tag pa1.

Two-class notMNIST dataset

The notMNIST dataset is a image recognition dataset of font glyphs for the letters A through J useful with simple neural networks. It is quite similar to the classic MNIST dataset of handwritten digits 0 through 9. We use the following script to generate a smaller dataset that only contains the images from two letter classes: \C”(the positive class) and \J”(the negative class). This smaller subset of the data contains 3500 training images, 100 validation images and 145 test images.

1 LOGISTIC REGRESSION WITH NUMPY[20 POINTS]


with np.load(’notMNIST.npz’) as data :

Data, Target = data [’images’], data[’labels’]

posClass = 2

negClass = 9

dataIndx = (Target==posClass) + (Target==negClass)

Data = Data[dataIndx]/255.

Target = Target[dataIndx].reshape(-1, 1)

Target[Target==posClass] = 1

Target[Target==negClass] = 0

np.random.seed(521)

randIndx = np.arange(len(Data))

np.random.shuffle(randIndx)

Data, Target = Data[randIndx], Target[randIndx] trainData, trainTarget = Data[:3500], Target[:3500] validData, validTarget = Data[3500:3600], Target[3500:3600] testData, testTarget = Data[3600:], Target[3600:]

Logistic Regression with Numpy[20 points]

Logistic regression is one the most widely used linear classi cation models in machine learning. In logistic regression, we model the probability of a sample x belonging to the positive class as

y^(x) = (w>x + b);

where z = w>x + b, also called logit, is basically the linear transformation of input vector x using weight vector w and bias scalar b. (z) = 1=(1 + exp( z)) is the sigmoid or logistic function: it \squashes” the real-valued logits to fall between zero and one.

The cross-entropy loss LCE and the regularization term Lw will form the total loss function as:

L =LCE + Lw

1 N

y(n) log y^

x(n)

1 y(n)

log 1

y^ x(n)

+

=N n=1

2 kwk22

X

Note that y(n) 2 f0; 1g is the class label for the n-th training image and is the regularization parameter.

1 LOGISTIC REGRESSION WITH NUMPY[20 POINTS]

Note: For part 1 of the assignment, you are not allowed to use TensorFlow or PyTorch. Your implementations should solely be based on Numpy.

Loss Function and Gradient [8 pts]:

Implement two vectorized Numpy functions (i.e. avoid using for loops by employing matrix products and broadcasting) to compute the loss function and its gradient. The grad_loss function should compute and return an analytical expression of the gradient of the loss with respect to both the weights and bias. Both function headers are below. Include the analytical expressions in your report as well as a snippet of your Python code.

def loss(w, b, x, y, reg): #Your implementation

def grad_loss(w, b, x, y, reg): #Your implementation

Gradient Descent Implementation [6 pts]:

Using the gradient computed from part 1, implement the batch Gradient Descent algorithm to classify the two classes in the notMNIST dataset. The function should accept 8 arguments – the weight vector, the bias, the data matrix, the labels, the learning rate, the number of epochs1, and an error tolerance (set to 1 10 7). The training should stop if the total number of epochs is reached, or the norm of the di erence between the old and updated weights are smaller than the error tolerance. The function should return the optimized weight vector and bias. The function header looks like the following:

def grad_descent(w, b, x, y, alpha, epochs, reg, error_tol):

#Your implementation here#

You may also wish to print and/or store the training, validation, and test losses/accuracies in this function for plotting. (In this case, you can add more inputs for validation and test data to your functions).

Tuning the Learning Rate[3 pts]:

Test your implementation of Gradient Descent with 5000 epochs and = 0. Investigate the impact of learning rate, = f0:005; 0:001; 0:0001g on the performance of your classi er. Plot the training and validation loss (on one gure) vs. number of passed epochs for each value of . Repeat this for training and validation accuracy. You should submit a total of 6 gures in your report for this part. Also, explain how you choose the best learning rate, and what accuracy you report for the selected learning rate.

Generalization [3 pts]:

Investigate the impact of regularization by modifying the regularization parameter, = f0:001; 0:1; 0:5g for = 0:005. Plot the training/validation loss/accuracy vs. epochs, similar

1Epoch is de ned as a complete pass of the training data. By de nition, batch gradient descent operates on the entire training dataset

2 LOGISTIC REGRESSION IN TENSORFLOW [20 POINTS]

to the previous part. Also, explain how you choose the best parameter, and what accuracy you report for the selected model.

Logistic Regression in TensorFlow [20 points]

In the exercises above, you implemented the Batch Gradient Descent Algorithm. For large datasets however, obtaining the loss gradient using all the training data at each iteration may be infeasible. Stochastic Gradient Descent, or Mini-batch gradient descent is aimed at solving this problem. You will be implementing the SGD algorithm and optimizing the training process using the Adaptive Moment Estimation technique (Adam), using TensorFlow.

Building the Computational Graph [5 pts]:

De ne a function, buildGraph() that initializes the TensorFlow computational graph. To do so, you must initialize the following:

The weight and bias tensors: for the weight tensors, use tf.truncated_normal and set the standard deviation to 0.5. Initial the bias variable with zero.

Placeholders for data, labels and : use tf.placeholder.

The loss tensor: Calculates the CE loss function with the regularization term. You may wish to investigate the TensorFlow API Documentation regarding losses and regular-ization on their website.

The optimizer: use tf.train.AdamOptimizer to minimize the total loss. Set the learning rate to 0.001.

The function should return the TensorFlow objects for weight, bias, predicted labels, real labels, the loss, and the optimizer.

Note: If you are using TF2 or PyTorch, you won’t need placeholders at all. In this case, build_graph() will have a misleading name, since graphs are not built explicitly in TF2. Instead, you need a similar function that only performs the forward pass of your model: it takes inputs and generates outputs.

Implementing Stochastic Gradient Descent [5 pts.] For the training loop, implement the SGD algorithm using a minibatch size of 500 optimizing over 700 epochs 2. Calculate the total number of batches required by dividing the number of training instances by the minibatch size. After each epoch you will need to reshu e the training data and start sampling from the beginning again. Initially, set = 0 and continue to use the same value (i.e. 0.001). After each epoch, store the training and validation losses and accuracies. Use these to plot the loss and accuracy curves.

2An epoch refers to a complete pass of the training data. SGD makes weight updates based on a sample of the training data.

LOGISTIC REGRESSION IN TENSORFLOW [20 POINTS]

Batch Size Investigation [4 pts.] Study the e ects of batch size on behaviour of Adam by optimizing the model using batch sizes of B = f100; 700; 1750g. Also, set = 0 and continue to use = 0:001. For each batch size, plot training/validation loss in one plot and training/validation accuracy in another plot (you need to have a total of 6 plots for this section). What is the impact of batch size on the nal classi cation accuracy for each of the 3 cases? Can you justify this observation?

Hyperparameter Investigation [4 pts.] Experiment with the following Adam hyperpa-rameters and for each, report on the nal training, validation and test accuracies. Explain which value you pick for hyperparameters in each part, and what accuracy you report.

1 = f0:95; 0:99g

2 = f0:99; 0:9999g

(c) = f1e 09; 1e 4g

For this part, use a minibatch size B = 500, a learning rate of = 0:001 with no regular-ization, and optimize over 700 epochs. For each of the three hyperparameters listed above, keep the other two as the default TensorFlow initialization. Note that in order to set 1, 2, and , you may wish to add these parameters to build_graph() inputs.

Comparison against Batch GD [2 pts.] Comment on the overall performance of the SGD algorithm with Adam vs. the batch gradient descent algorithm you implemented earlier by comparing plots of the losses and accuracies of the Adam vs. batch gradient descent.
