from __future__ import division, print_function, absolute_import
from utils.utils import initial_imputation, normalization, renormalization, onehot_decoding, onehot_encoding
from tqdm import tqdm

import tensorflow as tf
import numpy as np

def autoencoder_imputation(data_x, data_m, cat_index, num_index, all_levels, DAE_params, num_imputations):
    no, dim = data_x.shape
    # initial imputation
    data_x = initial_imputation(data_x, cat_index, num_index)

    data_train = np.array([])
    data_train_m = np.array([])
    ## encode cat
    if cat_index:
        data_cat = data_x[:, cat_index]
        data_cat_m = data_m[:, cat_index]
        data_cat_enc, data_cat_enc_miss = onehot_encoding(data_cat, data_cat_m, all_levels, has_miss=False)
        n_classes = list(map(lambda x: len(x), all_levels))
        data_train = data_cat_enc
        data_train_m = data_cat_enc_miss
    ## normalize num
    if num_index:
        data_num = data_x[:, num_index]
        data_num_m = data_m[:, num_index]
        data_num_norm, norm_parameters = normalization(data_num)
        data_train = np.concatenate([data_train, data_num_norm], axis=1) if data_train.size else data_num_norm
        data_train_m = np.concatenate([data_train_m, data_num_m], axis=1) if data_train_m.size else data_num_m

    # Training Parameters
    learning_rate = DAE_params["learning_rate"]
    num_steps1 = DAE_params["num_steps_phase1"]
    num_steps2 = DAE_params["num_steps_phase2"]
    batch_size = DAE_params["batch_size"]

    # Network Parameters
    num_input = data_train.shape[1]
    num_hidden_1 = data_train.shape[1] + DAE_params["theta"]  # 1st layer num features
    num_hidden_2 = data_train.shape[1] + 2 * DAE_params["theta"]  # 2nd layer num features (the latent dim)
    num_hidden_3 = data_train.shape[1] + 3 * DAE_params["theta"]

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
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x_noise, weights['encoder_h1']),
                                       biases['encoder_b1']))
        layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                       biases['encoder_b3']))
        return layer_3

    # Building the decoder
    @tf.function
    def decoder(x):
        layer_1 = tf.nn.tanh(tf. add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        layer_3 = tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                         biases['decoder_b3'])
        col_index = 0
        empty_G_out = True
        # apply softmax to each categorical variable
        if cat_index:
            empty_G_out = False
            output = tf.nn.softmax(layer_3[:, :n_classes[0]])
            col_index = n_classes[0]
            for j in range(1, len(n_classes)):
                output = tf.concat(values=[output, tf.nn.softmax(layer_3[:, col_index:col_index + n_classes[j]])], axis=1)
                col_index += n_classes[j]
        # apply sigmoid to all numerical variables
        if num_index:
            out_num = tf.nn.sigmoid(layer_3[:, col_index:])
            output = tf.concat(values=[output, out_num], axis=1) if not empty_G_out else out_num
        return output

    # sum up loss for each categorical variable
    @tf.function
    def dae_loss(y_pred, y_true, mask):
        loss = 0
        current_ind = 0
        # categorical loss
        if cat_index:
            for j in range(len(n_classes)):
                mask_current = mask[:, current_ind:current_ind + n_classes[j]]
                y_pred_current = y_pred[:, current_ind:current_ind + n_classes[j]]
                y_true_current = y_true[:, current_ind:current_ind + n_classes[j]]
                loss += -tf.reduce_mean(
                    input_tensor=mask_current * y_true_current * tf.math.log(mask_current * y_pred_current + 1e-8)) / tf.reduce_mean(
                    input_tensor=mask_current)
                current_ind += n_classes[j]
        # numerical loss
        if num_index:
            mask_current = mask[:, current_ind:]
            y_pred_current = y_pred[:, current_ind:]
            y_true_current = y_true[:, current_ind:]
            loss += tf.reduce_mean((mask_current * y_true_current - mask_current * y_pred_current)**2) / tf.reduce_mean(mask_current)
        return loss

    # optimizer
    @tf.function
    def optimize_step(batch_x, batch_m):
        with tf.GradientTape() as g:
            y_hat = decoder(encoder(batch_x))
            l = dae_loss(y_hat, batch_x, batch_m)

        trainable_variables = list(weights.values()) + list(biases.values())

        gradients = g.gradient(l, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        return l, y_hat

    optimizer = tf.optimizers.Adam(lr=learning_rate, decay=0.0)

    # multiple imputation
    imputed_list = []
    for l in range(num_imputations):
        # Start Training
        # Training phase 1
        loss_list = []
        pbar = tqdm(range(num_steps1))
        for i in pbar:
            # create mini batch
            indices = np.arange(no)
            np.random.shuffle(indices)
            for start_idx in range(0, no - batch_size + 1, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                batch_x = data_train[batch_idx, :]
                batch_m = data_train_m[batch_idx, :]

                # Run optimization op (backprop) and cost op (to get loss value)
                l, y_hat = optimize_step(batch_x, batch_m)
                pbar.set_description("loss at epoch {}: {:.3f}, phase 1".format(i, l))
                loss_list.append(l)

        imputed_data = decoder(encoder(data_train))
        imputed_data = data_train_m * data_train + (1 - data_train_m) * imputed_data

        # Training phase 2
        pbar = tqdm(range(num_steps2))
        for i in pbar:
            # create mini batch
            indices = np.arange(no)
            np.random.shuffle(indices)
            for start_idx in range(0, no - batch_size + 1, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                batch_x = tf.gather(imputed_data, batch_idx, axis=0)
                batch_m = data_train_m[batch_idx, :]

                # Run optimization op (backprop) and cost op (to get loss value)
                l, y_hat = optimize_step(batch_x, batch_m)
                pbar.set_description("loss at epoch {}, phase 2: {:.3f}".format(i, l))
                loss_list.append(l)

        # get imputation
        imputed_data = decoder(encoder(imputed_data))
        imputed_data = data_train_m * data_train + (1 - data_train_m) * imputed_data

        # revert onehot and renormalize
        imputed = np.empty(shape=(no, dim))
        if cat_index:
            imputed_cat = imputed_data[:, :data_cat_enc.shape[1]]
            imputed_cat = onehot_decoding(imputed_cat, data_cat_enc_miss, all_levels, has_miss=False)
            imputed[:, cat_index] = imputed_cat
        if num_index:
            imputed_num = imputed_data[:, -data_num.shape[1]:]
            imputed_num = renormalization(imputed_num.numpy(), norm_parameters)
            imputed[:, num_index] = imputed_num
        imputed_list.append(imputed)
    return imputed_list, loss_list