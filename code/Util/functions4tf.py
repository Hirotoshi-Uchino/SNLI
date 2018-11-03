import tensorflow as tf

def batch_norm_wrapper(inputs, is_training, decay = 0.999):
    epsilon = 1.e-8
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), name='scale')
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), name='beta')
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False, name='pop_mean')
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False, name='pop_var')

    if is_training is True:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

def prelu(x, alpha):
    return tf.maximum(tf.zeros(tf.shape(x)), x) + \
                      alpha*tf.minimum(tf.zeros(tf.shape(x)), x)


def make_batch_indexes(tgt_size, batch_size):
    batch_indexes = []
    n_batches = tgt_size // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch_indexes.append({'start': start, 'end': end})
    batch_indexes.append({'start': end, 'end': tgt_size})

    return batch_indexes
