# Author: Jonathan Siegel
#
# Gives a simple example of a generative model based upon invertible flow which generates samples from the MNIST dataset.

# Load the training and test data.
import jax.numpy as jnp
from matplotlib import pyplot
import os
import pickle
import optax

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

key = random.PRNGKey(0)

# Fetch full datasets for evaluation
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
# You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
mnist_data = tfds.as_numpy(mnist_data)
train_data, test_data = mnist_data['train'], mnist_data['test']
h, w, c = info.features['image'].shape
num_pixels = h * w * c

# Full train set
train_images, train_labels = train_data['image'], train_data['label']
train_images = jnp.moveaxis(train_images * (1.0/256), -1, 1)

# Full test set
test_images, test_labels = test_data['image'], test_data['label']
test_images = jnp.moveaxis(test_images * (1.0/256), -1, 1)

if __name__ == '__main__':
  # Show one of the images
  if input('Display some MNIST images? (yes/no): ') == 'yes':
    for i in range(10):
      pyplot.imshow(test_images[i,0,:,:])
      pyplot.show()

# A function to randomly initialize weights and biases
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Generates parameters for a single network.
def generate_net_parameters(m, n, d, key):
  keys = random.split(key, d)
  params = []
  params.append(random_layer_params(m, n, keys[0], scale = (2.0 / m)))
  for i in range(d-2):
    params.append(random_layer_params(n, n, keys[i+1], scale = (2.0 / n)))
  params.append(random_layer_params(n, m, keys[-1], scale = (2.0 / n)))
  return params

# Generates parameters for a single block.
def generate_block_parameters(m, n, d, key):
  return generate_net_parameters(m, n, d, key)

# Generates the full set of network parameters.
def generate_parameters(m, n, d, blocks, key):
  keys = random.split(key, blocks)
  params = []
  for i in range(blocks):
    params.append(generate_block_parameters(m, n, d, keys[i]))
  return params

def relu(x):
  return jnp.maximum(0,x)

# Applies the forward pass for a single block.
def apply_block_forward(input, block_params, resolution, parity):
  size1 = input.shape[1]
  size2 = input.shape[2]
  x,y = jnp.meshgrid(jnp.arange(size1), jnp.arange(size2))
  mask1 = ((x//resolution + y//resolution + parity)%2 == 0)
  mask2 = ((x//resolution + y//resolution + parity)%2 == 1)
  # Apply the shift network
  shift = (input * mask1).flatten()
  for w, b in block_params[:-1]:
    shift = jnp.dot(w, shift)
    shift = shift + b.reshape(shift.shape)
    shift = relu(shift)

  final_w, final_b = block_params[-1]
  shift = jnp.dot(final_w, shift) + final_b
  shift = shift.reshape(1, size1, size2)

  # Return the output of the block.
  return input + shift * mask2

# Applies the backward pass for a single block.
def apply_block_backward(input, block_params, resolution, parity):
  size1 = input.shape[1]
  size2 = input.shape[2]
  x,y = jnp.meshgrid(jnp.arange(size1), jnp.arange(size2))
  mask1 = ((x//resolution + y//resolution + parity)%2 == 0)
  mask2 = ((x//resolution + y//resolution + parity)%2 == 1)
  # Apply the shift network
  shift = (input * mask1).flatten()
  for w, b in block_params[:-1]:
    shift = jnp.dot(w, shift)
    shift = shift + b.reshape(shift.shape)
    shift = relu(shift)

  final_w, final_b = block_params[-1]
  shift = jnp.dot(final_w, shift) + final_b
  shift = shift.reshape(1, size1, size2)

  # Return the output of the block.
  return input - shift * mask2

batched_block_forward = vmap(apply_block_forward, in_axes=(0, None, None, None))
batched_block_backward = vmap(apply_block_backward, in_axes=(0, None, None, None))

# Applies the entire forward pass
def apply_forward_pass(input, params, resolutions, parities):
  for param, resolution, parity in zip(params, resolutions, parities):
    input = batched_block_forward(input, param, resolution, parity)
  # Map images from [-1,1]^n to [0,1]^n
  input = (input + 1.0) / 2
  return input

# Applies the entire backward pass.
def apply_backward_pass(input, params, resolutions, parities):
  # Map images from [0,1]^n to [-1,1]^n.
  input = 2 * input - 1.0
  for param, resolution, parity in zip(reversed(params), reversed(resolutions), reversed(parities)):
    input = batched_block_backward(input, param, resolution, parity)
  return input

if __name__ == '__main__':
  # Model hyperparameters
  m = 28 * 28
  n = 50 * 50
  d = 3
  blocks = 22
  resolutions = [14, 14, 7, 7, 14, 14, 7, 7, 14, 14, 7, 7, 14, 14, 7, 7, 14, 14, 7, 7, 14, 14]
  parities = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  artificial_noise_level = 0.05

  params = None
  if os.path.isfile(save_name):
    if input('Found a trained model. Load it (yes/no): ') == 'yes':
      f = open(save_name, 'rb')
      params = pickle.load(f)
      f.close()

  if params == None:
    print('Generating random initial parameters.')
    params = generate_parameters(m, n, d, blocks, key)

  if input('Test numerical invertibility (yes/no): ') == 'yes':
    input_images = train_images[0:10,:,:,:] 
    output = apply_forward_pass(input_images, params, resolutions, parities)
    new_input = apply_backward_pass(output, params, resolutions, parities)
    err = jnp.max(jnp.abs(input_images - new_input))
    print('Invertibility Error: ', err)

  # Define the negative log likelihood loss function.
  def loss(params, images):
    preimages = apply_backward_pass(images, params, resolutions, parities)
    loss = (1.0 / (2.0 * images.shape[0])) * jnp.sum(preimages * preimages)
    return loss

  # Train the network.
  if input('Train model (yes/no): ') == 'yes':
    iterations = 10000
    batch_size = 60
    lr = 1e-4
    # mom = 0.9
    optimizer = optax.radam(learning_rate = lr) # optimizer = optax.sgd(learning_rate = lr, momentum = mom)
    opt_state = optimizer.init(params)

    keys = random.split(key, iterations)
    for i in range(iterations):
      inds = random.choice(keys[i], train_images.shape[0], [batch_size])
      images = train_images[inds,:,:,:]
      images = images + artificial_noise_level * random.normal(keys[i], images.shape)
      print(i, loss(params, images))
      grads = grad(loss)(params, images)
      updates, opt_state = optimizer.update(grads, opt_state)
      params = optax.apply_updates(params, updates)

  if input('Generate Images (yes/no): ') == 'yes':
    num_images = int(input('Number of Images: '))
    rand_inputs = 1.1 * artificial_noise_level * random.normal(key, (num_images, 1, 28, 28))
    out_images = apply_forward_pass(rand_inputs, params, resolutions, parities)
    # Show the images
    for i in range(num_images):
      pyplot.imshow(out_images[i,0,:,:])
      pyplot.show()

  if input('Save model (yes/no): ') == 'yes':
    f = open(save_name, 'wb')
    pickle.dump(params, f)
    f.close()
