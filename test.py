import tensorflow as tf

# Make multilayer perceptron

weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 128], stddev=0.1)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([256, 10], stddev=0.1))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([256], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([10], stddev=0.1))
}


def mlp(_X, _W, _b, _keepprob):
    # Reshape input
    _input_r = tf.reshape(_X, shape=[-1, 28, 28, 1])
    # Conv1
    _conv1 = tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(_input_r, _W['wc1'], strides=[1, 1, 1, 1], padding='SAME')
            , _b['bc1'])
    )
    # Pool1
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # DropOut1
    _layer1 = tf.nn.dropout(_pool1, _keepprob)
    # Conv2
    _conv2 = tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(_layer1, _W['wc2'], strides=[1, 1, 1, 1], padding='SAME')
            , _b['bc2'])
    )
    # Pool2 (Global average pooling)
    print(_conv2)
    _pool2 = tf.nn.avg_pool(_conv2, ksize=[1, 14, 14, 1], strides=[1, 14, 14, 1], padding='SAME')
    print(_pool2)
    # DropOut2
    _layer2 = tf.nn.dropout(_pool2, _keepprob)
    print(_layer2)
    # Vectorize
    _dense = tf.reshape(_layer2, [-1, _W['out'].get_shape().as_list()[0]])
    # FC1
    _out = tf.nn.softmax(tf.add(tf.matmul(_dense, _W['out']), _b['out']))
    out = {
        'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'layer1': _layer1,
        'conv2': _conv2, 'pool2': _pool2, 'layer2': _layer2, 'dense': _dense,
        'out': _out
    }
    return out


# Define Parameter
learning_rate = 0.001
training_epochs = 10
batch_size = 100
display_step = 1

# Define Functions
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keepprob = tf.placeholder(tf.float32)
pred = mlp(x, weights, biases, keepprob)['out']
