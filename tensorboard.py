import tensorflow as tf

#with tf.name_scope("MyProgram"):
#  with tf.name_scope("Group_A"):

a = tf.add(1, 2, name="add1_2")
b = tf.multiply(a, 3, name="mula_3")

#   with tf.name_scope("Group_B"):
c = tf.add(4, 5)
d = tf.multiply(c, 6)

#with tf.name_scope("Group_C"):
e = tf.multiply(4, 5)
f = tf.div(c, 6)
g = tf.add(b, d)
h = tf.multiply(g, f)

s=tf.Session()
print(s.run(h))

writer = tf.summary.FileWriter("output", s.graph)
print(s.run(h))
writer.close()
	
	
	
	
	
	
	