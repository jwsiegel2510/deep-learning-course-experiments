# Author: Jonathan Siegel
#
# Uses an invertible flow based generative model trained on MNIST combined with an MNIST classifier to generate class-conditional samples.

import jax.numpy as jnp
from matplotlib import pyplot
import os
import pickle
import math

from jax._src.api import jit
from jax import random
from jax import vmap
from jax import lax
from jax import grad

# Load the classifier and generator models

from cnn_mnist_classification import batched_predict
from invertible_flow_mnist_generative import apply_forward 

classifier_file = 'mnist_classifier.dat'
generator_file = 'mnist_generative_flow_model.dat'

f = open(classifier_file, 'rb')
classifier_params = pickle.load(f)
f.close()

f = open(generator_file, 'rb')
generator_params = pickle.load(f)
f.close

# Generate random initial guess
label = int(input('Image Label: '))
num_images = int(input('Number of Images: '))
channel_size = 16
inputs = random.normal(random.PRNGKey(0), (num_images, channel_size, 7, 7))

def generate_label_score(inputs, generator_params, classifier_params):
  classifier_output = batched_predict(classifier_params, apply_forward(inputs, generator_params))
  normalization = jnp.sum(jnp.exp(classifier_output), -1)
  return jnp.sum(jnp.log(normalization)) - jnp.sum(classifier_output, 0)[label]

steps = 200
step_size = 0.025
keys = random.split(random.PRNGKey(0), steps)
for i in range(steps):
  print(i)
  # Take an MCMC step based upon corrected stochastic gradient flow.
  label_score = generate_label_score(inputs, generator_params, classifier_params)
  grads = grad(generate_label_score)(inputs, generator_params, classifier_params)
  
  # Generate trial point
  trial_point = inputs - step_size * (grads + inputs) + math.sqrt(step_size) * random.normal(keys[i], inputs.shape)
  
  # Accept or Reject
  inputs = trial_point

out_images = apply_forward(inputs, generator_params)[:,0,:,:]
# Show the images
for i in range(num_images):
  pyplot.imshow(out_images[i,:,:])
  pyplot.show()

