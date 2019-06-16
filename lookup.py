import tensorflow as tf;  
import numpy as np;  
  
c = np.random.random([10,2])  
b = tf.nn.embedding_lookup(c, [1, 3])  
  
with tf.Session() as sess:  
    sess.run(tf.initialize_all_variables())  
    print sess.run(b)  
    print c  
