# global-average-pooling
Global Average Pooling Implemented in TensorFlow

At this point, **this repository is in development**. <br>
I made ResNet with global average pooling instead of traditional fully-connected layer.  <br>
But the model will be replaced by simpler model for you to understand GAP easily. 


### GAP Example Code

The input tensor to GAP is (4, 4, 128). <br>
In this example, I used 1 x 1 convolution to reduce filter size and then compute average pooling to 4 x 4 size. 
Lastly, the output tensor of the average pooling layer is flattened by tf.reduce_mean. 

The [Network In Network](https://arxiv.org/pdf/1312.4400.pdf) paper is not obvious. 
So.. please let me know if something is wrong. 

```
gap_filter = resnet.create_variable('filter', shape=(1, 1, 128, 10))
h = tf.nn.conv2d(h, filter=gap_filter, strides=[1, 1, 1, 1], padding='SAME')
h = tf.nn.avg_pool(h, ksize=[1, 4, 4, 256], strides=[1, 1, 1, 1], padding='VALID')
h = tf.reduce_mean(h, axis=[1, 2])
```
