from __future__ import division, print_function, absolute_import
from sklearn.base import BaseEstimator
from utils.utils import initial_imputation, normalization, renormalization, onehot_decoding, onehot_encoding
from tqdm import tqdm
from utils.utils import rmse_loss

import tensorflow as tf
import numpy as np

class MIDA_sklearn():
    # Building the encoder
    @tf.function
    def encoder(self, x):
        x_noise = tf.nn.dropout(x, 0.5)
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x_noise, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))
        layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, self.weights['encoder_h3']),
                                       self.biases['encoder_b3']))
        return layer_3

    # Building the decoder
    @tf.function
    def decoder(self, x, n_classes):
        layer_1 = tf.nn.tanh(tf. add(tf.matmul(x, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        layer_3 = tf.add(tf.matmul(layer_2, self.weights['decoder_h3']),
                         self.biases['decoder_b3'])
        col_index = 0
        empty_G_out = True
        # apply softmax to each categorical variable
        if self.cat_index:
            empty_G_out = False
            output = tf.nn.softmax(layer_3[:, :n_classes[0]])
            col_index = n_classes[0]
            for j in range(1, len(n_classes)):
                output = tf.concat(values=[output, tf.nn.softmax(layer_3[:, col_index:col_index + n_classes[j]])], axis=1)
                col_index += n_classes[j]
        # apply sigmoid to all numerical variables
        if self.num_index:
            out_num = tf.nn.sigmoid(layer_3[:, col_index:])
            output = tf.concat(values=[output, out_num], axis=1) if not empty_G_out else out_num
        return output

    # sum up loss for each categorical variable
    @tf.function
    def dae_loss(self, y_pred, y_true, mask, n_classes):
        loss = 0
        current_ind = 0
        # categorical loss
        if self.cat_index:
            for j in range(len(n_classes)):
                mask_current = mask[:, current_ind:current_ind + n_classes[j]]
                y_pred_current = y_pred[:, current_ind:current_ind + n_classes[j]]
                y_true_current = y_true[:, current_ind:current_ind + n_classes[j]]
                loss += -tf.reduce_mean(
                    input_tensor=mask_current * y_true_current * tf.math.log(mask_current * y_pred_current + 1e-8)) / (tf.reduce_mean(
                    input_tensor=mask_current))
                current_ind += n_classes[j]
        # numerical loss
        if self.num_index:
            mask_current = mask[:, current_ind:]
            y_pred_current = y_pred[:, current_ind:]
            y_true_current = y_true[:, current_ind:]
            loss += tf.reduce_mean((mask_current * y_true_current - mask_current * y_pred_current)**2) / tf.reduce_mean(mask_current)
        return loss

    # optimizer
    @tf.function
    def optimize_step(self, batch_x, batch_m, n_classes):
        with tf.GradientTape() as g:
            y_hat = self.decoder(self.encoder(batch_x), n_classes)
            l = self.dae_loss(y_hat, batch_x, batch_m, n_classes)

        trainable_variables = list(self.weights.values()) + list(self.biases.values())

        gradients = g.gradient(l, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return l, y_hat

    def __init__(self, num_base_nodes, cat_index, num_index, all_levels,
                 learning_rate, num_steps_phase1, num_steps_phase2, batch_size, theta):
        self.cat_index = cat_index
        self.num_index = num_index
        self.all_levels = all_levels

        # Training Parameters
        learning_rate = learning_rate
        self.num_steps1 = num_steps_phase1
        self.num_steps2 = num_steps_phase2
        self.batch_size = batch_size

        # Network Parameters
        num_input = num_base_nodes
        num_hidden_1 = num_base_nodes + theta  # 1st layer num features
        num_hidden_2 = num_base_nodes + 2 * theta  # 2nd layer num features (the latent dim)
        num_hidden_3 = num_base_nodes + 3 * theta

        # A random value generator to initialize weights.
        random_normal = tf.initializers.RandomNormal()

        self.weights = {
            'encoder_h1': tf.Variable(random_normal([num_input, num_hidden_1])),
            'encoder_h2': tf.Variable(random_normal([num_hidden_1, num_hidden_2])),
            'encoder_h3': tf.Variable(random_normal([num_hidden_2, num_hidden_3])),
            'decoder_h1': tf.Variable(random_normal([num_hidden_3, num_hidden_2])),
            'decoder_h2': tf.Variable(random_normal([num_hidden_2, num_hidden_1])),
            'decoder_h3': tf.Variable(random_normal([num_hidden_1, num_input])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(random_normal([num_hidden_1])),
            'encoder_b2': tf.Variable(random_normal([num_hidden_2])),
            'encoder_b3': tf.Variable(random_normal([num_hidden_3])),
            'decoder_b1': tf.Variable(random_normal([num_hidden_2])),
            'decoder_b2': tf.Variable(random_normal([num_hidden_1])),
            'decoder_b3': tf.Variable(random_normal([num_input])),
        }

        self.optimizer = tf.optimizers.Adam(lr=learning_rate, decay=0.0)

    def preprocess_data(self, data_x, data_m):
        data_train = np.array([])
        data_train_m = np.array([])
        ## encode cat
        if self.cat_index:
            data_cat = data_x[:, self.cat_index]
            data_cat_m = data_m[:, self.cat_index]
            data_cat_enc, data_cat_enc_miss = onehot_encoding(data_cat, data_cat_m, self.all_levels, has_miss=False)
            n_classes = list(map(lambda x: len(x), self.all_levels))
            data_train = data_cat_enc
            data_train_m = data_cat_enc_miss
        ## normalize num
        if self.num_index:
            data_num = data_x[:, self.num_index]
            data_num_m = data_m[:, self.num_index]
            data_num_norm, norm_parameters = normalization(data_num)
            data_train = np.concatenate([data_train, data_num_norm], axis=1) if data_train.size else data_num_norm
            data_train_m = np.concatenate([data_train_m, data_num_m], axis=1) if data_train_m.size else data_num_m

        return data_train, data_train_m, n_classes, data_cat_enc, data_cat_enc_miss, data_num, norm_parameters

    # let X be data_m, y be data_x
    def fit(self, X, y=None, X_test = None, y_test = None, check=False):
        data_x = X
        data_m = 1 - np.isnan(data_x).astype(np.float32)
        no, dim = data_x.shape
        # initial imputation
        data_x = initial_imputation(data_x, self.cat_index, self.num_index)

        data_train, data_train_m, n_classes, data_cat_enc, data_cat_enc_miss, data_num, norm_parameters = self.preprocess_data(data_x, data_m)

        # check model fitting
        if check:
            train_loss = []
            test_loss = []

        # Train
        # Start Training
        # Training phase 1
        loss_list = []
        pbar = tqdm(range(self.num_steps1))
        for i in pbar:
            # create mini batch
            indices = np.arange(no)
            np.random.shuffle(indices)
            for start_idx in range(0, no - self.batch_size + 1, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                batch_x = data_train[batch_idx, :]
                batch_m = data_train_m[batch_idx, :]

                # Run optimization op (backprop) and cost op (to get loss value)
                l, y_hat = self.optimize_step(batch_x, batch_m, n_classes)
                pbar.set_description("loss at epoch {}: {:.3f}, phase 1".format(i, l))
                loss_list.append(l)

                if check:
                    data_test_m = 1 - np.isnan(X_test).astype(np.float32)
                    y_pred = self.predict(X)
                    y_test_pred = self.predict(X_test)
                    train_loss.append(rmse_loss(y, y_pred, data_m))
                    test_loss.append(rmse_loss(y_test, y_test_pred, data_test_m))

        if check:
            return train_loss, test_loss
        pass

    def predict(self, X):
        data_x = X
        data_m = 1 - np.isnan(data_x).astype(np.float32)
        no, dim = data_x.shape

        # initial imputation
        data_x = initial_imputation(data_x, self.cat_index, self.num_index)

        data_train, data_train_m, n_classes, data_cat_enc, data_cat_enc_miss, data_num, norm_parameters = self.preprocess_data(data_x, data_m)

        imputed_data = self.decoder(self.encoder(data_train), n_classes)
        imputed_data = data_train_m * data_train + (1 - data_train_m) * imputed_data

        # Training phase 2
        for i in range(self.num_steps2):
            # create mini batch
            indices = np.arange(no)
            np.random.shuffle(indices)
            for start_idx in range(0, no - self.batch_size + 1, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                batch_x = tf.gather(imputed_data, batch_idx, axis=0)
                batch_m = data_train_m[batch_idx, :]

                # Run optimization op (backprop) and cost op (to get loss value)
                l, y_hat = self.optimize_step(batch_x, batch_m, n_classes)

        # get imputation
        imputed_data = self.decoder(self.encoder(imputed_data), n_classes)
        imputed_data = data_train_m * data_train + (1 - data_train_m) * imputed_data

         # revert onehot and renormalize
        imputed = np.empty(shape=(no, dim))
        if self.cat_index:
            imputed_cat = imputed_data[:, :data_cat_enc.shape[1]]
            imputed_cat = onehot_decoding(imputed_cat, data_cat_enc_miss, self.all_levels, has_miss=False)
            imputed[:, self.cat_index] = imputed_cat
        if self.num_index:
            imputed_num = imputed_data[:, -data_num.shape[1]:]
            imputed_num = renormalization(imputed_num.numpy(), norm_parameters)
            imputed[:, self.num_index] = imputed_num
        return imputed