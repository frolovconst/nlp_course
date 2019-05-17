import numpy as np
import tensorflow as tf

def my_rnn(x_emb, emb_size, hid_size):
    """ takes x_emb[time, batch, emb_size] and predicts"""
    W = tf.Variable(np.random.randn(emb_size + hid_size, hid_size).astype('float32'),)
    h0 = tf.zeros([tf.shape(x_emb)[1], hid_size])
    
    def scan_step(h_t, x_t):
      rnn_inp = tf.concat([h_t, x_t], axis=1)
      h_next = tf.tanh(tf.matmul(x_t, W))
      return h_next
      

    return tf.scan(scan_step, elems=x_emb, initializer=h0)