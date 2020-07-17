'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index


def gain (data_x, data_m, n_classes, gain_parameters):
  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  
  # Other parameters
  no, dim = data_x.shape

  # Hidden state dimensions
  h_Gdim = 512
  h_Ddim = int(dim)

  data_x = np.nan_to_num(data_x, 0)

  ## GAIN architecture
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_Ddim])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_Ddim]))
  
  D_W2 = tf.Variable(xavier_init([h_Ddim, h_Ddim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_Ddim]))
  
  D_W3 = tf.Variable(xavier_init([h_Ddim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_Gdim]))
  G_b1 = tf.Variable(tf.zeros(shape = [h_Gdim]))
  
  G_W2 = tf.Variable(xavier_init([h_Gdim, h_Gdim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_Gdim]))

  G_W3 = tf.Variable(xavier_init([h_Gdim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  @tf.function
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1)
    G_h1 = tf.nn.leaky_relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.leaky_relu(tf.matmul(G_h1, G_W2) + G_b2)

    # MinMax normalized output
    G_prob = tf.matmul(G_h2, G_W3) + G_b3
    G_out = tf.nn.softmax(G_prob[:, :n_classes[0]])
    col_index = n_classes[0]
    for j in range(1, len(n_classes)):
      G_out = tf.concat(values=[G_out, tf.nn.softmax(G_prob[:, col_index:col_index + n_classes[j]])], axis=1)
      col_index += n_classes[j]
    return G_out
  # Discriminator
  @tf.function
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1)
    D_h1 = tf.nn.leaky_relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.leaky_relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob

  # loss function
  @tf.function
  def gain_Dloss(D_prob, mask):
    D_loss_temp = -tf.reduce_mean(mask * tf.math.log(D_prob + 1e-7) +
                                  (1 - mask) * tf.math.log(1. - D_prob + 1e-7))
    D_loss = D_loss_temp
    return D_loss

  @tf.function
  def gain_Gloss(sample, G_sample, D_prob, mask, n_classes):
    G_loss_temp = -tf.reduce_mean((1 - mask) * tf.math.log(D_prob + 1e-7))

    MSE_loss = 0
    current_ind = 0
    for j in range(len(n_classes)):
      M_current = mask[:, current_ind:current_ind + n_classes[j]]
      G_sample_temp = G_sample[:, current_ind:current_ind + n_classes[j]]
      X_temp = sample[:, current_ind:current_ind + n_classes[j]]
      MSE_loss += -tf.reduce_mean(M_current * X_temp * tf.math.log(M_current * G_sample_temp + 1e-7)) / tf.reduce_mean(
        M_current)
      current_ind += n_classes[j]

    G_loss = G_loss_temp + alpha * MSE_loss

    return G_loss

  # optimizer
  @tf.function
  def optimize_step(X_mb, M_mb, H_mb, n_classes):
    with tf.GradientTape() as g:
      # Generator
      G_sample = generator(X_mb, M_mb)
      # Combine with observed data
      Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
      # Discriminator
      D_prob = discriminator(Hat_X, H_mb)
      D_loss = gain_Dloss(D_prob, M_mb)

    Dgradients = g.gradient(D_loss, theta_D)
    D_solver.apply_gradients(zip(Dgradients, theta_D))

    with tf.GradientTape() as g:
      # Generator
      G_sample = generator(X_mb, M_mb)
      # Combine with observed data
      Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
      # Discriminator
      D_prob = discriminator(Hat_X, H_mb)
      G_loss = gain_Gloss(X_mb, G_sample, D_prob, M_mb, n_classes)
    Ggradients = g.gradient(G_loss, theta_G)
    G_solver.apply_gradients(zip(Ggradients, theta_G))
    return D_loss, G_loss

  ## GAIN solver
  D_solver = tf.optimizers.Adam()
  G_solver = tf.optimizers.Adam()


  # Start Iterations
  pbar = tqdm(range(iterations))
  for it in pbar:
      
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = data_x[batch_idx, :]  
    M_mb = data_m[batch_idx, :]

    # Sample random vectors  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp

    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

    D_loss_curr, G_loss_curr = optimize_step(X_mb, M_mb, H_mb, n_classes)

    pbar.set_description("D_loss: {:.3f}, G_loss: {:.3f}".format(D_loss_curr, G_loss_curr))

  ## Return imputed data
  Z_mb = uniform_sampler(0, 0.01, no, dim) 
  M_mb = data_m
  X_mb = data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
  imputed_data = generator(X_mb, M_mb)
  
  imputed_data = data_m * data_x + (1-data_m) * imputed_data
          
  return imputed_data