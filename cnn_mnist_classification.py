# Contains a simple example in which a small CNN is trained to classify MNIST.

import jax.numpy as jnp
from matplotlib import pyplot
import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')
import os
import pickle

import tensorflow_datasets as tfds
from jax._src.api import jit
from jax import random
from jax import vmap
from jax import lax
from jax import grad

if __name__ == '__main__':
  save_name = './mnist_classifier.dat'

  # Load the training and test data.
  data_dir = './mnist_data'

  def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

  # Fetch full datasets for evaluation
  # tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
  # You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
  mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
  mnist_data = tfds.as_numpy(mnist_data)
  train_data, test_data = mnist_data['train'], mnist_data['test']
  num_labels = info.features['label'].num_classes
  h, w, c = info.features['image'].shape
  num_pixels = h * w * c

  # Full train set
  train_images, train_labels = train_data['image'], train_data['label']
  train_images = jnp.moveaxis(train_images * (1.0/256), -1, 1)
  train_labels = one_hot(train_labels, num_labels)

  # Full test set
  test_images, test_labels = test_data['image'], test_data['label']
  test_images = jnp.moveaxis(test_images * (1.0/256), -1, 1)
  test_labels = one_hot(test_labels, num_labels)

  # Show one of the images
  pyplot.imshow(test_images[0,0,:,:])
  pyplot.show()

# A helper function to randomly initialize weights and biases
def random_layer_params(m, n, key):
  w_key, b_key = random.split(key)
  scale = 2.0 / jnp.sqrt(m)
  return scale * random.normal(w_key, (m, n)), scale * random.normal(b_key, (n,))

# A helper function to randomly initialize weights and biases for the convolutional layers
# m: input channels
# n: output channels
# k: kernel size
def random_conv_layer_params(m, n, k, key):
  w_key, b_key = random.split(key)
  scale = 2.0 / (k * jnp.sqrt(m))
  return scale * random.normal(w_key, (n, m, k, k)), scale * random.normal(b_key, (n,))

# Initialize all layers for a CNN with the given block structure.
def init_network_params(conv_blocks, full_widths, key):
  keys = random.split(key, 2)
  conv_keys = random.split(keys[0], len(conv_blocks))
  conv_layers = []
  for i in range(len(conv_blocks)):
    block_keys = random.split(conv_keys[i], len(conv_blocks[i]) - 1)
    channels = conv_blocks[i]
    conv_layers.append([random_conv_layer_params(m,n,5,key) for m,n,key in zip(channels[:-1], channels[1:], block_keys)])
  full_keys = random.split(keys[1], len(full_widths) - 1)
  full_layer = [random_layer_params(m,n,key) for m,n,key in zip(full_widths[:-1], full_widths[1:], full_keys)]
  return [conv_layers, full_layer]

# Activation functions
def sigmoid(x):
  return 1.0 / (1.0 + jnp.exp(-x))

def relu(x):
  return jnp.maximum(0,x)

# Apply the convolutional layer
@jit
def apply_conv_layer(conv_layer, images):
  for conv_weights, biases in conv_layer[:-1]:
      images = lax.conv(images, conv_weights, (1,1), 'SAME')
      images = relu(images +
                    # Interesting hack to get bias to right shape.
                    jnp.moveaxis(jnp.broadcast_to(biases, [images.shape[0], images.shape[2], images.shape[3], images.shape[1]]), -1,1))
  conv_weights, biases = conv_layer[-1]
  # Convolution with stride to downsample the image
  images = lax.conv(images, conv_weights, (2,2), 'SAME')
  return relu(images + jnp.moveaxis(jnp.broadcast_to(biases, [images.shape[0], images.shape[2], images.shape[3], images.shape[1]]), -1,1))

# Calculate output based upon network parameters
@jit
def batched_predict(params, images):
  [conv_layers, full_layer] = params
  for blocks in conv_layers:
    images = apply_conv_layer(blocks, images)
  images = jnp.reshape(images, [images.shape[0],-1])
  for w,b in full_layer[:-1]:
    images = jnp.dot(images, w)
    images = images + b
    images = relu(images)
  final_w, final_b = full_layer[-1]
  return jnp.dot(images, final_w) + final_b

# Enable evaluation at multiple points simultaneously
# batched_predict = vmap(predict, in_axes=(None, 0))

if __name__ == '__main__':
  params = init_network_params([[1,5],[5,20],[20,50]], [800,200,10], random.PRNGKey(0))

from jax import grad

if __name__ == '__main__':
  #  Define the Cross Entropy loss for classification.
  @jit
  def loss(params, images, labels):
    output = batched_predict(params, images)
    normalization = jnp.sum(jnp.exp(output), -1)
    return (1.0/images.shape[0]) * (jnp.sum(jnp.log(normalization)) - jnp.sum(jnp.multiply(output, labels)))

  # Define the gradient descent update step.
  def update(params, images, labels, step, momentum, velocity):
    grads = grad(loss)(params, images, labels)
    if velocity == None:
      velocity = grads
    else:
      conv_velocity = [[(momentum * w + dw, momentum * b + db) for (w, b), (dw, db) in zip(velocity_blocks, block_grad)]
                       for velocity_blocks, block_grad in zip(velocity[0], grads[0])]
      dense_velocity = [(momentum * w + dw, momentum * b + db)
                        for (w, b), (dw, db) in zip(velocity[1], grads[1])]
      velocity = [conv_velocity, dense_velocity]
    new_conv_layers = [[(w - step * dw, b - step * db) for (w, b), (dw, db) in zip(blocks, block_velocity)]
                       for blocks, block_velocity in zip(params[0], velocity[0])]
    new_dense_layers = [(w - step * dw, b - step * db)
                        for (w, b), (dw, db) in zip(params[1], velocity[1])]
    return velocity, [new_conv_layers, new_dense_layers]

  # Train the neural network
  def train(params, train_images, train_labels, num_steps, batch_size, step_size, momentum, key):
    keys = random.split(key, num_steps)
    velocity = None
    for i in range(num_steps):
      print('Step : ', i)
      inds = random.choice(keys[i], train_images.shape[0], [batch_size])
      step_images = train_images[inds,:,:,:]
      step_labels = train_labels[inds,:,]
      velocity, params = update(params, step_images, step_labels, step_size, momentum, velocity)
    return params

  num_steps = 4000
  batch_size = 50
  step_size = 0.04
  momentum = 0.8

  params = train(params, train_images, train_labels, num_steps, batch_size, step_size, momentum, random.PRNGKey(0))

  # Calculate test accuracy.
  def test_accuracy(params, test_images, test_labels):
    predictions = jnp.argmax(batched_predict(params, test_images), -1)
    correct = 0.0
    for i in range(predictions.shape[0]):
      correct = correct + test_labels[i][predictions[i]]
    return correct / predictions.shape[0]

  print(test_accuracy(params, test_images, test_labels))

  f = open(save_name, 'wb')
  pickle.dump(params, f)
  f.close()
