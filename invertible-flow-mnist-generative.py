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

# Generates random parameters for the invertible block.
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

# Generates parameters which specify the per channel scaling.
# m: number of channels
def generate_scaling_parameters(m, key, scale = 0.0):
  return scale * random.normal(key, (m,))

# Generates the full set of network parameters.
# m: half of the number of channels
# n: number of inner residual channels
# d: inner residual depth
# k: kernel size
# l: number of blocks
def generate_parameters(m, n, d, k, l, key):
  parameters = []
  keys = random.split(key, l)
  for i in range(l):
    block_key, scale_key = random.split(keys[i], 2)
    block_params = generate_block_parameters(m, n, d, k, block_key)
    scaling_params = generate_scaling_parameters(2*m, scale_key)
    permutation = jnp.array([2*j + (i%2) for j in range(m)] + [2*j + (i+1)%2 for j in range(m)])
    parameters.append([block_params, scaling_params, permutation])
  return parameters

# Activation functions
def sigmoid(x):
  return 1.0 / (1.0 + jnp.exp(-x))

def relu(x):
  return jnp.maximum(0,x)

# Applies the forward pass. Specifically, applies the block to the channels
# specified in channels_in and adds the result to the channels in channels_add.
def apply_block_forward(input, block_params, channels_in, channels_add, step = 1.0):
  conv_weights, biases = block_params[0]
  residual = lax.conv(input[:,channels_in,:,:], conv_weights, (1,1), 'SAME')
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
  residual = lax.conv(input[:,channels_in,:,:], conv_weights, (1,1), 'SAME')
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

# Applies the per channel scaling.
def apply_scaling_forward(input, scaling_params, factor = 0.1):
  return input * jnp.moveaxis(jnp.broadcast_to(jnp.exp(factor * scaling_params), [input.shape[0], input.shape[2], input.shape[3], input.shape[1]]), -1,1)

def apply_scaling_backward(input, scaling_params, factor = 0.1):
  return input * jnp.moveaxis(jnp.broadcast_to(jnp.exp(-factor * scaling_params), [input.shape[0], input.shape[2], input.shape[3], input.shape[1]]), -1,1)

# Define a function squeezing the input to [0,1] and its inverse.
def squeeze_function(x):
  return 0.5 + (1/jnp.pi) * jnp.arctan(x)

def squeeze_inverse(x):
  return jnp.tan(jnp.pi * (x - 0.5))

channel_size = 16
sqrtchannels = 4
factor = 0.1

# Applies the full network forward pass.
@jit
def apply_forward(input, parameters):
  for block_params, scaling_params, permutation in parameters:
    m = len(permutation)
    input = apply_block_forward(input, block_params, permutation[:m//2], permutation[m//2:], step = 0.1)
    input = apply_scaling_forward(input, scaling_params, factor)
  output = jnp.zeros((input.shape[0], 1, sqrtchannels * input.shape[2], sqrtchannels * input.shape[3]))
  for i in range(channel_size):
    output = output.at[:,0,(i%sqrtchannels) * input.shape[2]:(i%sqrtchannels + 1) * input.shape[2],(i//sqrtchannels) * input.shape[3]:(i//sqrtchannels + 1) * input.shape[3]].set(input[:,i,:,:])
  # Apply a function which squeezes the first input channel to [0,1] to obtain an image.
  output = squeeze_function(output)
  return output

# Applies the full network backward pass.
@jit
def apply_backward(input, parameters):
  input = input.at[:,0,:,:].apply(squeeze_inverse)
  output = jnp.zeros((input.shape[0], channel_size, input.shape[2]//sqrtchannels, input.shape[3]//sqrtchannels))
  for i in range(channel_size):
    output = output.at[:,i,:,:].set(input[:,0,(i%sqrtchannels) * output.shape[2]:(i%sqrtchannels + 1) * output.shape[2],(i//sqrtchannels) * output.shape[3]:(i//sqrtchannels + 1) * output.shape[3]])
  for block_params, scaling_params, permutation in reversed(parameters):
    m = len(permutation)
    output = apply_scaling_backward(output, scaling_params, factor)
    output = apply_block_backward(output, block_params, permutation[:m//2], permutation[m//2:], step = 0.1)
  return output

inner_channels = 64
kernel_size = 5
inner_depth = 3
layers = 8

parameters = None
if os.path.isfile(save_name):
  if input('Found a trained model. Load it (yes/no): ') == 'yes':
    f = open(save_name, 'rb')
    parameters = pickle.load(f)
  else:
    parameters = generate_parameters(channel_size // 2, inner_channels, inner_depth, kernel_size, layers, random.PRNGKey(0))

if parameters == None:
  print('No saved models found. Generating random initial parameters.')
  parameters = generate_parameters(channel_size // 2, inner_channels, inner_depth, kernel_size, layers, random.PRNGKey(0))

if input('Test numerical invertibility (yes/no): ') == 'yes':
  rand_in = random.normal(random.PRNGKey(0), (1,16,7,7))
  output = apply_forward(rand_in, parameters)
  new_in = apply_backward(output, parameters)
  err = jnp.max(jnp.abs(rand_in - new_in))
  print('Invertibility Error: ', err) 

# Define the negative log likelihood loss function.
def loss(params, images):
  preimages = apply_backward(images, params)
  preimage_loss = (1.0 / (2.0 * images.shape[0])) * jnp.sum(preimages * preimages)
  transformation_loss = 0.0
  for block_params, scaling_params, permutation in params:
    transformation_loss = transformation_loss + factor * jnp.sum(scaling_params)
  return preimage_loss + transformation_loss

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
def train(params, train_images, num_steps, batch_size, step_size, momentum, key):
  keys = random.split(key, num_steps)
  velocity = None
  for i in range(num_steps):
    print(i)
    inds = random.choice(keys[i], train_images.shape[0], [batch_size])
    step_images = 0.9 * train_images[inds,:,:,:] + 0.05 # scale so that the pixels are strictly between [0,1]
    velocity, params = update(params, step_images, step_size, momentum, velocity)
  return params

num_steps = 3000
batch_size = 50
step_size = 0.00000025
momentum = 0.9

if input('Train model (yes/no): ') == 'yes':
  parameters = train(parameters, train_images, num_steps, batch_size, step_size, momentum, random.PRNGKey(0))

if input('Generate Images (yes/no): ') == 'yes':  
  num_images = int(input('Number of Images: '))
  rand_inputs = random.normal(random.PRNGKey(0), (num_images, channel_size, 7, 7))
  out_images = apply_forward(rand_inputs, parameters)[:,0,:,:]
  # Show the images
  for i in range(num_images):
    pyplot.imshow(out_images[i,:,:])
    pyplot.show()

if input('Save model (yes/no): ') == 'yes':
  f = open(save_name, 'wb')
  pickle.dump(parameters, f)
