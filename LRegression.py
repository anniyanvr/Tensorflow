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