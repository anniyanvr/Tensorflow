import tensorflow as tf
a = tf.constant([1,2,3,4,5,6], shape=[2,3])
b= tf.constant([7,8,9,10,11,12], shape=[3,2])

product = tf.matmul(a,b)
sum = tf.add(a,tf.transpose(b))

s=tf.Session()

print("\n\n")
print(s.run(a))

print("\n\n")
print(s.run(b))

print("\n\n")
print(s.run(product))

print("\n\n")
print(s.run(sum))

s.close()