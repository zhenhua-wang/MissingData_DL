from __future__ import division, print_function, absolute_import
from utils import sample_batch_index
from tqdm import tqdm

import tensorflow as tf
import numpy as np

def autoencoder_imputation(data_x, data_m, n_classes, model_params, train_params):
    no, dim = data_x.shape

    # Training Parameters
    learning_rate = train_params["learning_rate"]
    num_steps1 = train_params["num_steps_phase1"]
    num_steps2 = train_params["num_steps_phase2"]
    batch_size = train_params["batch_size"]

    # Network Parameters
    num_input = dim
    num_hidden_1 = dim + model_params["theta"] # 1st layer num features
    num_hidden_2 = dim + 2*model_params["theta"] # 2nd layer num features (the latent dim)
    num_hidden_3 = dim + 3*model_params["theta"]

    # A random value generator to initialize weights.
    random_normal = tf.initializers.RandomNormal()

    weights = {
        'encoder_h1': tf.Variable(random_normal([num_input, num_hidden_1])),
        'encoder_h2': tf.Variable(random_normal([num_hidden_1, num_hidden_2])),
        'encoder_h3': tf.Variable(random_normal([num_hidden_2, num_hidden_3])),
        'decoder_h1': tf.Variable(random_normal([num_hidden_3, num_hidden_2])),
        'decoder_h2': tf.Variable(random_normal([num_hidden_2, num_hidden_1])),
        'decoder_h3': tf.Variable(random_normal([num_hidden_1, num_input])),
    }
    biases = {
        'encoder_b1': tf.Variable(random_normal([num_hidden_1])),
        'encoder_b2': tf.Variable(random_normal([num_hidden_2])),
        'encoder_b3': tf.Variable(random_normal([num_hidden_3])),
        'decoder_b1': tf.Variable(random_normal([num_hidden_2])),
        'decoder_b2': tf.Variable(random_normal([num_hidden_1])),
        'decoder_b3': tf.Variable(random_normal([num_input])),
    }

    # Building the encoder
    @tf.function
    def encoder(x):
        x_noise = tf.nn.dropout(x, 0.5)
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x_noise, weights['encoder_h1']),
                                       biases['encoder_b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                       biases['encoder_b3']))
        return layer_3


    # Building the decoder
    @tf.function
    def decoder(x):
        layer_1 = tf.nn.relu(tf. add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        layer_3 = tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                         biases['decoder_b3'])
        # apply softmax to each categorical variable
        output = tf.nn.softmax(layer_3[:, :n_classes[0]])
        col_index = n_classes[0]
        for j in range(1, len(n_classes)):
            output = tf.concat(values=[output, tf.nn.softmax(layer_3[:, col_index:col_index + n_classes[j]])], axis=1)
            col_index += n_classes[j]
        return output

    # sum up loss for each categorical variable
    @tf.function
    def cat_loss(y_pred, y_true, mask, n_classes):
        loss = 0
        current_ind = 0
        for j in range(len(n_classes)):
            mask_current = mask[:, current_ind:current_ind + n_classes[j]]
            y_pred_current = y_pred[:, current_ind:current_ind + n_classes[j]]
            y_true_current = y_true[:, current_ind:current_ind + n_classes[j]]
            loss += -tf.reduce_mean(
                input_tensor=mask_current * y_true_current * tf.math.log(mask_current * y_pred_current + 1e-8)) / tf.reduce_mean(
                input_tensor=mask_current)
            current_ind += n_classes[j]
        return loss

    # optimizer
    @tf.function
    def optimize_step(batch_x, batch_m, n_classes):
        with tf.GradientTape() as g:
            y_hat = decoder(encoder(batch_x))
            l = cat_loss(y_hat, batch_x, batch_m, n_classes)

        trainable_variables = list(weights.values()) + list(biases.values())

        gradients = g.gradient(l, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        return l, y_hat

    optimizer = tf.optimizers.SGD(lr=learning_rate, momentum=0.99, nesterov=True)

    # Start Training
    loss_list = []
    # Training phase 1
    pbar = tqdm(range(num_steps1))
    for i in pbar:
        # Sample batch
        batch_idx = sample_batch_index(no, batch_size)
        batch_x = data_x[batch_idx, :]
        batch_m = data_m[batch_idx, :]

        # Run optimization op (backprop) and cost op (to get loss value)
        l, y_hat = optimize_step(batch_x, batch_m, n_classes)
        pbar.set_description("loss: {:.3f}".format(l))
        loss_list.append(l)
    imputed_data = decoder(encoder(data_x))
    imputed_data = data_m * data_x + (1 - data_m) * imputed_data

    # phase 2
    for i in tqdm(range(num_steps2)):
        imputed_data = decoder(encoder(imputed_data))
        imputed_data = data_m * data_x + (1 - data_m) * imputed_data

    return imputed_data, loss_list