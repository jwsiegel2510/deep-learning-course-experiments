# Contains a simple example in which an MLP is trained to classify MNIST.

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
  save_name = './mnist_mlp_classifier.dat'

  ### Load the training and test data.
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

  # Show one of the images
  pyplot.imshow(test_images[1,0,:,:])
  pyplot.show()

  ### Construct and train the network.
  input_dim = 784
  output_dim = 10 # Number of classes
  width = 100
  depth = 4
  
  # A helper function to randomly initialize weights and biases
  def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

  def init_deep_network_params(n, d, input_dim, output_dim, scale, key):
    keys = random.split(key, d+1)
    sizes = [input_dim]
    for i in range(d):
      sizes.append(n)
    sizes.append(output_dim)
    return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

  def relu(x):
    return jnp.maximum(0,x)

  # Calculate output based upon network parameters
  def predict(params, values):
    # Flatten input image into a vector
    values = values.flatten()
    activations = values
    for w, b in params[:-1]:
      outputs = jnp.dot(w, activations)
      outputs = outputs + b.reshape(outputs.shape)
      activations = relu(outputs)

    final_w, final_b = params[-1]
    return jnp.dot(final_w, activations) + final_b

  # Enable evaluation at multiple points simultaneously
  batched_predict = vmap(predict, in_axes=(None, 0))

  # Initialize the network parameters
  params = init_deep_network_params(width, depth, input_dim, output_dim, 1.0/jnp.sqrt(width), random.PRNGKey(0))

  # Define the Cross Entropy loss function
  @jit
  def loss(params, images, labels):
    output = batched_predict(params, images)
    normalization = jnp.sum(jnp.exp(output), -1)
    return (1.0/images.shape[0]) * (jnp.sum(jnp.log(normalization)) - jnp.sum(jnp.multiply(output, labels)))

  # Train the network using gradient descent
  @jit
  def update(params, images, labels, step):
    grads = grad(loss)(params, images, labels)
    return [(w - step * dw, b - step * db)
            for (w, b), (dw, db) in zip(params, grads)]

  step_size = 0.05
  batch_size = 50
  num_steps = 4000

  # Train the neural network
  def train(params, train_images, train_labels, num_steps, batch_size, step_size, key):
    keys = random.split(key, num_steps)
    for i in range(num_steps):
      print('Step : ', i)
      inds = random.choice(keys[i], train_images.shape[0], [batch_size])
      step_images = train_images[inds,:,:,:]
      step_labels = train_labels[inds,:,]
      params = update(params, step_images, step_labels, step_size)
    return params

  params = train(params, train_images, train_labels, num_steps, batch_size, step_size, random.PRNGKey(0))

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
