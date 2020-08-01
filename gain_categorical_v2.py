'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm

from utils import xavier_init
from utils import binary_sampler, uniform_sampler, onehot, normalization, reverse_onehot, renormalization

def gain (data_x, data_m, cat_index, num_index, all_levels, gain_parameters):
    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    data_cat, data_num = data_x[:, cat_index], data_x[:, num_index]
    data_cat_m, data_num_m = data_m[:, cat_index], data_m[:, num_index]
    # preprocess categorical variables
    data_cat = np.nan_to_num(data_cat, 0)
    data_cat_enc, data_cat_enc_miss, n_classes, onehot_enc = onehot(data_cat, data_cat_m, all_levels)
    # preprocess numerical variables
    data_num_norm, norm_parameters = normalization(data_num)
    data_num_norm = np.nan_to_num(data_num_norm, 0)
    # concatenate to training data
    data_train = np.concatenate([data_cat_enc, data_num_norm], axis=1)
    data_train_m = np.concatenate([data_cat_enc_miss, data_num_m], axis=1)

    # Other parameters
    no, dim = data_x.shape
    n_classes = list(map(lambda x: len(x), all_levels))
    input_dim = data_train.shape[1]

    # Hidden state dimensions
    h_Gdim = gain_parameters['h_Gdim']
    h_Ddim = gain_parameters['h_Ddim']

    ## GAIN architecture
    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([input_dim*2, h_Ddim])) # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape = [h_Ddim]))

    D_W2 = tf.Variable(xavier_init([h_Ddim, h_Ddim]))
    D_b2 = tf.Variable(tf.zeros(shape = [h_Ddim]))

    D_W3 = tf.Variable(xavier_init([h_Ddim, input_dim]))
    D_b3 = tf.Variable(tf.zeros(shape = [input_dim]))  # Multi-variate outputs

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    #Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([input_dim*2, h_Gdim]))
    G_b1 = tf.Variable(tf.zeros(shape = [h_Gdim]))

    G_W2 = tf.Variable(xavier_init([h_Gdim, h_Gdim]))
    G_b2 = tf.Variable(tf.zeros(shape = [h_Gdim]))

    G_W3 = tf.Variable(xavier_init([h_Gdim, input_dim]))
    G_b3 = tf.Variable(tf.zeros(shape = [input_dim]))

    theta_G = [G_W1, G_W3, G_b1, G_b3]

    ## GAIN functions
    # Generator
    @tf.function
    def generator(x,m):
        temperature = 0.5
        # Concatenate Mask and Data
        inputs = tf.concat(values = [x, m], axis = 1)
        G_h1 = tf.nn.leaky_relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.leaky_relu(tf.matmul(G_h1, G_W2) + G_b2)

        # apply softmax to each categorical variable
        G_logit = tf.matmul(G_h2, G_W3) + G_b3
        G_out = tf.nn.softmax(G_logit[:, :n_classes[0]])
        # dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=G_prob[:, :n_classes[0]])
        # G_out = dist.probs_parameter()
        col_index = n_classes[0]
        for j in range(1, len(n_classes)):
            # dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=G_prob[:, col_index:col_index + n_classes[j]])
            # G_out = tf.concat(values=[G_out, dist.probs_parameter()], axis=1)
            G_out = tf.concat(values=[G_out, tf.nn.softmax(G_logit[:, col_index:col_index + n_classes[j]])], axis=1)
            col_index += n_classes[j]
        # apply sigmoid to all numerical variables
        G_out = tf.concat(values=[G_out, tf.nn.sigmoid(G_logit[:, col_index:])], axis=1)
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
        # categorical loss
        reconstruct_loss = 0
        current_ind = 0
        for j in range(len(n_classes)):
            M_current = mask[:, current_ind:current_ind + n_classes[j]]
            G_sample_temp = G_sample[:, current_ind:current_ind + n_classes[j]]
            X_temp = sample[:, current_ind:current_ind + n_classes[j]]
            reconstruct_loss += -tf.reduce_mean(M_current * X_temp * tf.math.log(M_current * G_sample_temp + 1e-7)) / tf.reduce_mean(
                M_current)
            current_ind += n_classes[j]
        # numerical loss
        M_current = mask[:, current_ind:]
        G_sample_temp = G_sample[:, current_ind:]
        X_temp = sample[:, current_ind:]
        reconstruct_loss += tf.reduce_mean((M_current * X_temp - M_current * G_sample_temp) ** 2) / tf.reduce_mean(
            M_current)

        return G_loss_temp, reconstruct_loss

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

        for i in range(3):
            with tf.GradientTape() as g:
                # Generator
                G_sample = generator(X_mb, M_mb)
                # Combine with observed data
                Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
                # Discriminator
                D_prob = discriminator(Hat_X, H_mb)
                G_loss_temp, reconstructloss = gain_Gloss(X_mb, G_sample, D_prob, M_mb, n_classes)
                G_loss = G_loss_temp + alpha*reconstructloss
            Ggradients = g.gradient(G_loss, theta_G)
            G_solver.apply_gradients(zip(Ggradients, theta_G))
        return D_loss, G_loss_temp, reconstructloss

    ## GAIN solver
    D_solver = tf.optimizers.Adam()
    G_solver = tf.optimizers.Adam()


    # Start Iterations
    Gloss_list = []
    Dloss_list = []
    pbar = tqdm(range(iterations))
    for i in pbar:
        # create mini batch
        indices = np.arange(no)
        np.random.shuffle(indices)
        for start_idx in range(0, no - batch_size + 1, batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            X_mb = data_train[batch_idx, :]
            M_mb = data_train_m[batch_idx, :]

            # Sample random vectors
            Z_mb = uniform_sampler(0, 0.01, batch_size, input_dim)
            # Sample hint vectors
            H_mb_temp = binary_sampler(hint_rate, batch_size, input_dim)
            H_mb = M_mb * H_mb_temp

            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

            D_loss_curr, G_loss_curr, reconstructloss = optimize_step(X_mb, M_mb, H_mb, n_classes)
            Gloss_list.append(G_loss_curr)
            Dloss_list.append(D_loss_curr)
            pbar.set_description("D_loss: {:.3f}, G_loss: {:.3f}, Reconstruction loss: {:.3f}".format(D_loss_curr.numpy(),
                                                                                                      G_loss_curr.numpy(),
                                                                                                      reconstructloss.numpy()))

    ## Return imputed data
    Z_mb = uniform_sampler(0, 0.01, no, input_dim)
    M_mb = data_train_m
    X_mb = data_train
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

    imputed_data = generator(X_mb, M_mb)
    imputed_data = data_train_m * data_train + (1-data_train_m) * imputed_data

    # revert onehot and renormalize
    imputed_cat, imputed_num = imputed_data[:, :data_cat_enc.shape[1]], imputed_data[:, data_cat_enc.shape[1]:]
    imputed_cat = reverse_onehot(imputed_cat, onehot_enc)
    imputed_num = renormalization(imputed_num.numpy(), norm_parameters)
    imputed = np.empty(shape=(no, dim))
    imputed[:, cat_index] = imputed_cat
    imputed[:, num_index] = imputed_num
    return imputed, Gloss_list, Dloss_list