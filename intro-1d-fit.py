# Contains a simple example using a small neural network to fit a 1d function with gradient descent.
import jax.numpy as jnp
from jax import random
from jax import vmap
import matplotlib.pyplot as plt
from jax._src.api import jit
from jax import grad

# A helper function to randomly initialize weights and biases
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a deep neural network with width n and depth d
def init_deep_network_params(n, d, scale, key):
  keys = random.split(key, d+1)
  sizes = [1]
  for i in range(d):
    sizes.append(n)
  sizes.append(1)
  return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# Activation functions
def sigmoid(x):
  return 1.0 / (1.0 + jnp.exp(-x))

def relu(x):
  return jnp.maximum(0,x)

# Calculate output based upon network parameters
def predict(params, values):
  activations = values
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations)
    outputs = outputs + b.reshape(outputs.shape)
    activations = relu(outputs)

  final_w, final_b = params[-1]
  return jnp.dot(final_w, activations) + final_b

# Enable evaluation at multiple points simultaneously
batched_predict = vmap(predict, in_axes=(None, 0))

# Initialize and evaluate the network
width = 20
depth = 3
init_scale = 1/jnp.sqrt(width) # For ReLU activation
# init_scale = 1 # For sigmoidal activation function
params = init_deep_network_params(width, depth, init_scale, random.PRNGKey(0))

# Plot the function at initialization
num_pts = 100
vals = jnp.linspace(-1,1,num_pts)
outs = batched_predict(params, vals).reshape([num_pts])
plt.plot(vals, outs)
plt.show()

# Generate random points in -1,1 and training data to fit sin.
def generate_train_data_sin(n):
  vals = jnp.linspace(-1,1,num_pts)
  outs = jnp.sin(jnp.pi * vals)
  return [vals, outs]

# Generate random points in -1,1 and training data to fit the Heaviside.
def generate_train_data_heaviside(n):
  vals = jnp.linspace(-1,1,num_pts)
  outs = (vals > 0)
  return [vals, outs]

# Define the l2 regression loss function/
def loss(params, vals, outs):
  preds = batched_predict(params, vals).reshape(-1,)
  return (1.0 / vals.size)*jnp.sum(jnp.multiply(preds - outs, preds - outs))

@jit
def update(params, vals, outs, step):
  grads = grad(loss)(params, vals, outs)
  return [(w - step * dw, b - step * db)
          for (w, b), (dw, db) in zip(params, grads)]

num_steps = 5000
data_size = 1000
step_size = 0.1
[vals, outs] = generate_train_data_sin(data_size)
for epoch in range(num_steps):
    params = update(params, vals, outs, step_size)

# print(params)
plt.plot(vals, outs)
outs = batched_predict(params, vals).reshape([num_pts])
plt.plot(vals, outs)
plt.show()
