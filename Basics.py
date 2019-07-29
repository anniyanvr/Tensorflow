#Program to demonstrate tf.constant, tf.Variable, tf.placeholder as well as tf.Session() 


import tensorflow as tf
'''
a = tf.constant(4.0, dtype=tf.float32)
b = tf.constant(6.0) # Also tf.float32 implicitly
sum = a + b
print("\n\n")
print(a)
print(b)
print(sum)

print("\n\n")
s = tf.Session()
print(s.run(a)) 
print(s.run(b)) 
print(s.run(sum))


x = tf.constant([[2, 2, 2], [1, 1, 1]])
print("\n\n")
#Computes the sum of elements across dimensions of a tensor
print(s.run(tf.reduce_sum(x)))               # 9 
print(s.run(tf.reduce_sum(x, 0)))            # [3, 3, 3]
print(s.run(tf.reduce_sum(x, 1)))            # [6, 3]
print(s.run(tf.reduce_sum(x, [0, 1])))       # 9
print(s.run(tf.reduce_prod(x, [0, 1])))      # 8                
'''

'''
x has a shape of (2, 3) (two rows and three columns):
By doing tf.reduce_sum(x, 0) the tensor is reduced along the first dimension
 (rows), so the result is [2, 2, 2] + [1, 1, 1] = [3, 3, 3].

By doing tf.reduce_sum(x, 1) the tensor is reduced along the second dimension
(columns), so the result is [2, 1] + [2, 1] + [2, 1] = [6, 3].

By doing tf.reduce_sum(x, [0, 1]) the tensor is reduced along BOTH dimensions
(rows and columns), so the result is [2, 2, 2] + [1, 1, 1] = [3, 3, 3], and then 3 + 3 + 3 = 9 
(reduce along rows, then reduce the resulted array).
'''

'''
s=tf.Session()
x=tf.Variable(5,name="x")
y=tf.Variable(2,name="y")
a=(x*x*y)
b=y+2
f=a+b
s.run(x.initializer)
s.run(y.initializer)
result=s.run(f)
print("\n\n")
print(result)
'''


x =tf.placeholder(tf.float32)
y =tf.placeholder(tf.float32)
z = x + y
sess = tf.Session()
print("\n\n")
print(sess.run(z,feed_dict={x:1, y:5.5}))
print(sess.run(z,feed_dict={x:[1,2], y:[3,4]}))