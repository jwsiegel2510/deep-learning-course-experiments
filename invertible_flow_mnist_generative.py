# Author: Jonathan Siegel
#
# Gives a simple example of a generative model based upon invertible flow which generates samples from the MNIST dataset.

# Load the training and test data.
import jax.numpy as jnp
from matplotlib import pyplot
import os
import pickle

from jax._src.api import jit
from jax import random
from jax import vmap
from jax import lax
from jax import grad

import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')

import tensorflow_datasets as tfds

data_dir = './mnist_data'

save_name = './mnist_generative_flow_model.dat'

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

if __name__ == '__main__':
  # Show one of the images
  if input('Display an MNIST image? (yes/no): ') == 'yes':
    pyplot.imshow(test_images[0,0,:,:])
    pyplot.show()

# A helper function to randomly initialize weights and biases for the convolutional layers
# m: input channels
# n: output channels
# k: kernel size
def random_conv_layer_params(m, n, k, key):
  w_key, b_key = random.split(key)
  scale = 2.0 / (k * jnp.sqrt(m))
  return [scale * random.normal(w_key, (n, m, k, k)), scale * random.normal(b_key, (n,))]

# Generates random parameters for the large scale invertible block.
# m: input and output channels
# n: inner channels
# d: depth
# k: kernel size
def generate_block_parameters(m, n, d, k, key):
  key_one, key = random.split(key)
  params = [random_conv_layer_params(m, n, k, key_one)]
  next_keys = random.split(key, d+1)
  for i in range(d):
    params.append(random_conv_layer_params(n, n, k, next_keys[i]))
  params.append(random_conv_layer_params(n, m, k, next_keys[d]))
  return params

# Generates random parameters for the small scale invertible block.
# n: inner channels
# d: depth
# k: kernel size
def generate_small_block_parameters(n, d, k, key):
  key_one, key = random.split(key)
  params = [random_conv_layer_params(1, n, k, key_one)]
  next_keys = random.split(key, d+1)
  for i in range(d):
    params.append(random_conv_layer_params(n, n, k, next_keys[i]))
  params.append(random_conv_layer_params(n, 1, k, next_keys[d]))
  return params

# Generates the full set of network parameters.
# m: half of the number of channels
# n1: number of inner residual channels
# d1: inner residual depth
# k1: kernel size
# l1: number of large scale blocks
# n2: number of small scale channels
# d2: small scale depth
# k2: small scale kernel size
# l2: number of small scale blocks
def generate_parameters(m, n1, d1, k1, l1, n2, d2, k2, l2, key):
  lsc_key, ssc_key = random.split(key)
  keys = random.split(lsc_key, l1)
  large_scale_params = []
  for i in range(l1):
    block_key, scale_key = random.split(keys[i], 2)
    block_params = generate_block_parameters(m, n1, d1, k1, block_key)
    permutation = jnp.array([2*j + (i%2) for j in range(m)] + [2*j + (i+1)%2 for j in range(m)])
    large_scale_params.append([block_params, permutation])
  keys = random.split(ssc_key, l2)
  small_scale_params = []
  for i in range(l2):
    block_params = generate_small_block_parameters(n2, d2, k2, keys[i])
    small_scale_params.append(block_params)
  return [large_scale_params, small_scale_params]

# Activation functions
def sigmoid(x):
  return 1.0 / (1.0 + jnp.exp(-x))

def relu(x):
  return jnp.maximum(0,x)

# Applies the forward pass. Specifically, applies the block to the channels
# specified in channels_in and adds the result to the channels in channels_add.
def apply_block_forward(input, block_params, channels_in, channels_add, step = 1.0):
  conv_weights, biases = block_params[0]
  residual = input[:,channels_in,:,:]
  residual = residual / (jnp.max(jnp.abs(residual)) + 1e-6)
  residual = lax.conv(residual, conv_weights, (1,1), 'SAME')
  residual = relu(residual +
                    # Interesting hack to get bias to right shape.
                    jnp.moveaxis(jnp.broadcast_to(biases, [residual.shape[0], residual.shape[2], residual.shape[3], residual.shape[1]]), -1,1))
  for conv_weights, biases in block_params[1:-1]:
    residual = lax.conv(residual, conv_weights, (1,1), 'SAME')
    residual = relu(residual +
                      # Interesting hack to get bias to right shape.
                      jnp.moveaxis(jnp.broadcast_to(biases, [residual.shape[0], residual.shape[2], residual.shape[3], residual.shape[1]]), -1,1))
  conv_weights, biases = block_params[-1]
  input = input.at[:,channels_add,:,:].add(step * (lax.conv(residual, conv_weights, (1,1), 'SAME') \
                                                        + jnp.moveaxis(jnp.broadcast_to(biases, [input.shape[0], input.shape[2], input.shape[3], len(channels_add)]), -1,1)))
  return input

# Applies the backward pass.
def apply_block_backward(input, block_params, channels_in, channels_add, step = 1.0):
  conv_weights, biases = block_params[0]
  residual = input[:,channels_in,:,:]
  residual = residual / (jnp.max(jnp.abs(residual)) + 1e-6)
  residual = lax.conv(residual, conv_weights, (1,1), 'SAME')
  residual = relu(residual +
                    # Interesting hack to get bias to right shape.
                    jnp.moveaxis(jnp.broadcast_to(biases, [residual.shape[0], residual.shape[2], residual.shape[3], residual.shape[1]]), -1,1))
  for conv_weights, biases in block_params[1:-1]:
    residual = lax.conv(residual, conv_weights, (1,1), 'SAME')
    residual = relu(residual +
                      # Interesting hack to get bias to right shape.
                      jnp.moveaxis(jnp.broadcast_to(biases, [residual.shape[0], residual.shape[2], residual.shape[3], residual.shape[1]]), -1,1))
  conv_weights, biases = block_params[-1]
  input = input.at[:,channels_add,:,:].add(-1.0 * step * (lax.conv(residual, conv_weights, (1,1), 'SAME') \
                                                        + jnp.moveaxis(jnp.broadcast_to(biases, [input.shape[0], input.shape[2], input.shape[3], len(channels_add)]), -1,1)))
  return input

# Applies the small scale forward pass.
def apply_forward_small_scale(input, block_params, parity, step = 1.0):
  conv_weights, biases = block_params[0]
  size1 = input.shape[2]
  size2 = input.shape[3]
  x,y = jnp.meshgrid(jnp.arange(size1), jnp.arange(size2))
  mask1 = ((x + y + parity)%2 == 0).astype(float)
  mask2 = ((x + y + parity)%2 == 1).astype(float)
  residual = input * mask1
  residual = residual / (jnp.max(jnp.abs(residual)) + 1e-6)
  residual = lax.conv(residual, conv_weights, (1,1), 'SAME')
  residual = relu(residual +
                    # Interesting hack to get bias to right shape.
                    jnp.moveaxis(jnp.broadcast_to(biases, [residual.shape[0], residual.shape[2], residual.shape[3], residual.shape[1]]), -1,1))
  for conv_weights, biases in block_params[1:-1]:
    residual = lax.conv(residual, conv_weights, (1,1), 'SAME')
    residual = relu(residual +
                      # Interesting hack to get bias to right shape.
                      jnp.moveaxis(jnp.broadcast_to(biases, [residual.shape[0], residual.shape[2], residual.shape[3], residual.shape[1]]), -1,1))
  conv_weights, biases = block_params[-1]
  input = input + step * mask2 * (lax.conv(residual, conv_weights, (1,1), 'SAME') \
                                                        + jnp.moveaxis(jnp.broadcast_to(biases, [input.shape[0], input.shape[2], input.shape[3], input.shape[1]]), -1,1))  
  return input

# Applies the small scale forward pass.
def apply_backward_small_scale(input, block_params, parity, step = 1.0):
  conv_weights, biases = block_params[0]
  size1 = input.shape[2]
  size2 = input.shape[3]
  x,y = jnp.meshgrid(jnp.arange(size1), jnp.arange(size2))
  mask1 = ((x + y + parity)%2 == 0)
  mask2 = ((x + y + parity)%2 == 1)
  residual = input * mask1
  residual = residual / (jnp.max(jnp.abs(residual)) + 1e-6)
  residual = lax.conv(residual, conv_weights, (1,1), 'SAME')
  residual = relu(residual +
                    # Interesting hack to get bias to right shape.
                    jnp.moveaxis(jnp.broadcast_to(biases, [residual.shape[0], residual.shape[2], residual.shape[3], residual.shape[1]]), -1,1))
  for conv_weights, biases in block_params[1:-1]:
    residual = lax.conv(residual, conv_weights, (1,1), 'SAME')
    residual = relu(residual +
                      # Interesting hack to get bias to right shape.
                      jnp.moveaxis(jnp.broadcast_to(biases, [residual.shape[0], residual.shape[2], residual.shape[3], residual.shape[1]]), -1,1))
  conv_weights, biases = block_params[-1]
  input = input - step * mask2 * (lax.conv(residual, conv_weights, (1,1), 'SAME') \
                                                        + jnp.moveaxis(jnp.broadcast_to(biases, [input.shape[0], input.shape[2], input.shape[3], input.shape[1]]), -1,1))  
  return input

# Define a function squeezing the input to [0,1] and its inverse.
def squeeze_function(x):
  return 0.5 + (1/jnp.pi) * jnp.arctan(x)

def squeeze_inverse(x):
  return jnp.tan(jnp.pi * (x - 0.5))

channel_size = 16
sqrtchannels = 4
image_size = 7
step = 1.0

# Applies the full network forward pass.
#@jit
def apply_forward(input, parameters):
  for block_params, permutation in parameters[0]:
    m = len(permutation)
    input = apply_block_forward(input, block_params, permutation[:m//2], permutation[m//2:], step)
  output = jnp.zeros((input.shape[0], 1, sqrtchannels * input.shape[2], sqrtchannels * input.shape[3]))
  for i in range(channel_size):
    output = output.at[:,0,(i%sqrtchannels) * input.shape[2]:(i%sqrtchannels + 1) * input.shape[2],(i//sqrtchannels) * input.shape[3]:(i//sqrtchannels + 1) * input.shape[3]].set(input[:,i,:,:])
  parity = 0
  for block_params in parameters[1]:
    output = apply_forward_small_scale(output, block_params, parity, step)
    parity = (parity + 1)%2
  # Apply a function which squeezes the first input channel to [0,1] to obtain an image.
  output = squeeze_function(output)
  return output

# Applies the full network backward pass.
#@jit
def apply_backward(input, parameters):
  input = input.at[:,0,:,:].apply(squeeze_inverse)
  parity = (len(parameters[1])+1)%2
  for block_params in reversed(parameters[1]):
    input = apply_backward_small_scale(input, block_params, parity, step)
    parity = (parity + 1)%2
  output = jnp.zeros((input.shape[0], channel_size, input.shape[2]//sqrtchannels, input.shape[3]//sqrtchannels))
  for i in range(channel_size):
    output = output.at[:,i,:,:].set(input[:,0,(i%sqrtchannels) * output.shape[2]:(i%sqrtchannels + 1) * output.shape[2],(i//sqrtchannels) * output.shape[3]:(i//sqrtchannels + 1) * output.shape[3]])
  for block_params, permutation in reversed(parameters[0]):
    m = len(permutation)
    output = apply_block_backward(output, block_params, permutation[:m//2], permutation[m//2:], step)
  return output

if __name__ == '__main__':
  inner_channels = 32
  kernel_size = 5
  inner_depth = 4
  layers = 8
  small_scale_inner_channels = 16
  small_scale_kernel_size = 5
  small_scale_inner_depth = 4
  small_scale_layers = 4

  parameters = None
  if os.path.isfile(save_name):
    if input('Found a trained model. Load it (yes/no): ') == 'yes':
      f = open(save_name, 'rb')
      parameters = pickle.load(f)
      f.close()
    else:
      parameters = generate_parameters(channel_size // 2, inner_channels, inner_depth, kernel_size, layers, 
                small_scale_inner_channels, small_scale_inner_depth, 
                small_scale_kernel_size, small_scale_layers, random.PRNGKey(0))

  if parameters == None:
    print('No saved models found. Generating random initial parameters.')
    parameters = generate_parameters(channel_size // 2, inner_channels, inner_depth, kernel_size, layers, 
                small_scale_inner_channels, small_scale_inner_depth, 
                small_scale_kernel_size, small_scale_layers, random.PRNGKey(0))


  if input('Test numerical invertibility (yes/no): ') == 'yes':
    rand_in = random.normal(random.PRNGKey(0), (1,16,7,7))
    output = apply_forward(rand_in, parameters)
    new_in = apply_backward(output, parameters)
    err = jnp.max(jnp.abs(rand_in - new_in))
    print('Invertibility Error: ', err) 

  # Define the negative log likelihood loss function.
  def loss(params, images):
    preimages = apply_backward(images, params)
    loss = (1.0 / (2.0 * images.shape[0])) * jnp.sum(preimages * preimages)
    return loss

  # Two functions which are called recursively for the update
  def rec_update(params, velocity, step):
    if type(params) is list:
      return [rec_update(p,v,step) for p,v in zip(params,velocity)]
    if str(params.dtype) == 'int32':
      return params
    return params - step * velocity

  def rec_update_vel(velocity, grads, momentum):
    if type(velocity) is list:
      return [rec_update_vel(v,g,momentum) for v,g in zip(velocity,grads)]
    if str(velocity.dtype) != 'float32':
      return velocity
    return momentum * velocity + grads

  # Define the stochastic gradient descent update step.
  def update(params, images, step, momentum, velocity):
    print(loss(params, images))
    grads = grad(loss, allow_int = True)(params, images)
    if velocity is None:
      velocity = grads
    else:
      velocity = rec_update_vel(velocity, grads, momentum)
    return velocity, rec_update(params, velocity, step)

  # Train the neural network
  def train(params, train_images, num_steps, batch_size, step_size, momentum, noise_level, key):
    keys = random.split(key, num_steps)
    velocity = None
    for i in range(num_steps):
      print(i)
      # Add random noise to regularize the empirical distribution.
      inds_keys, noise_keys = random.split(keys[i])
      inds = random.choice(inds_keys, train_images.shape[0], [batch_size])
      step_images = train_images[inds,:,:,:]
      step_images = jnp.clip(step_images + noise_level * random.normal(noise_keys, step_images.shape), 0, 1)
      step_images = 0.9 * step_images + 0.05 # scale so that the pixels are strictly between [0,1]
      velocity, params = update(params, step_images, step_size, momentum, velocity)
    return params

  num_steps = 5000
  batch_size = 10
  step_size = 0.000001
  momentum = 0.8
  noise_level = 0.03

  if input('Train model (yes/no): ') == 'yes':
    parameters = train(parameters, train_images, num_steps, batch_size, step_size, momentum, noise_level, random.PRNGKey(1))

  if input('Generate Images (yes/no): ') == 'yes':  
    num_images = int(input('Number of Images: '))
    # temperature = float(input('Temperature: '))
    rand_inputs = random.normal(random.PRNGKey(0), (num_images, channel_size, 7, 7))
    out_images = apply_forward(rand_inputs, parameters)[:,0,:,:]
    # Show the images
    for i in range(num_images):
      pyplot.imshow(out_images[i,:,:])
      pyplot.show()

  if input('Save model (yes/no): ') == 'yes':
    f = open(save_name, 'wb')
    pickle.dump(parameters, f)
    f.close()
