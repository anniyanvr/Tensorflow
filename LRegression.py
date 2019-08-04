#Import required modules for plotting linear regression.
import numpy as np 
import matplotlib.pyplot as p 

#Define the number of coefficients for logistic regression.
no_of_pnts = 100 
x_point = [] 
y_point = [] 
A = 0.3 
B = 0.7 

for i in range(no_of_pnts): 
   X = np.random.normal(0.0,0.5) 
   Y = A*X + B + np.random.normal(0.0,0.1) 
   x_point.append(X) 
   y_point.append(Y) 
   
#View the generated points using Matplotlib.
p.plot(x_point,y_point, 'o', label = 'Input Data') 
p.title('Logistic Regression')
p.xlabel('X Values')
p.ylabel('Y Values')
p.legend() 
p.show()


'''
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

#Define a seed that makes the random numbers predictable. 
#We will define fixed seeds for both Numpy and Tensorflow.
#When the value is reset, the same numbers will appear every time. 
#If we do not assign the seed, NumPy automatically selects a random seed value 
#based on the system's random number generator device or on the clock.

#tf.set_random_seed(101)

#Genrate random linear data...50 data points ranging from 0 to 50 
x = np.linspace(0, 50, 50) 
y = np.linspace(0, 50, 50) 

#Adding noise to random linear data
x += np.random.uniform(-4, 4, 50) 
y += np.random.uniform(-4, 4, 50) 

n = len(x) # Number of data points

#start creating our model by defining the placeholders X and Y, 
#so that we can feed our training examples X and Y into optimizer during the training process.
X = tf.placeholder(tf.float32) 
Y = tf.placeholder(tf.float32)

#Declare two trainable Tensorflow Variables for Weights and Bias
#Initialize them randomly using np.random.randn()
W = tf.Variable(np.random.randn(), name = "W") 
b = tf.Variable(np.random.randn(), name = "b")

#Define hyperparameters of the model: Learning Rate and Number of Epochs.
learning_rate = 0.01
training_epochs = 200

#Hypothesis 
y_pred = tf.add(tf.multiply(X, W), b) 

#Mean Squared Error Cost Function 
cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n)

#Gradient Descent Optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

#Global Variables Initializer 
init = tf.global_variables_initializer()

#Starting the Tensorflow Session 
with tf.Session() as sess: 
#Initializing the Variables 
    sess.run(init) 
	
#Iterating through all the epochs 
    for epoch in range(training_epochs): 
        
		#Feed each data point into the optimizer using Feed Dictionary 
        for (_x, _y) in zip(x, y): 
            sess.run(optimizer, feed_dict = {X : _x, Y : _y})
        
		#Display the result after every 50 epochs 
        if (epoch + 1) % 50 == 0: 
            #Calculate the cost every epoch 
            c = sess.run(cost, feed_dict = {X : x, Y : y}) 
            print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b))
    
	#Store necessary values to be used outside the Session 
    training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 
    weight = sess.run(W) 
    bias = sess.run(b)

predictions = weight * x + bias 
print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n')
plt.plot(x, y, 'ro', label ='Original data') 
plt.plot(x, predictions, label ='Fitted line') 
plt.title('Linear Regression Result') 
plt.legend() 
plt.show()

'''