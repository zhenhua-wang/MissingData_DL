import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator
from utils.utils import xavier_init, binary_sampler, uniform_sampler, onehot_encoding, onehot_decoding, normalization, renormalization
from utils.utils import rmse_loss

class GAIN_sklearn(BaseEstimator):
    ## GAIN functions
    # Generator
    @tf.function
    def generator(self, x, m, n_classes):
        # Concatenate Mask and Data
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.leaky_relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        G_h2 = tf.nn.leaky_relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        G_logit = tf.matmul(G_h2, self.G_W3) + self.G_b3

        col_index = 0
        empty_G_out = True
        # apply softmax to each categorical variable
        if self.cat_index:
            empty_G_out = False
            G_out = tf.nn.softmax(G_logit[:, :n_classes[0]])
            col_index = n_classes[0]
            for j in range(1, len(n_classes)):
                G_out = tf.concat(values=[G_out, tf.nn.softmax(G_logit[:, col_index:col_index + n_classes[j]])], axis=1)
                col_index += n_classes[j]
        # apply sigmoid to all numerical variables
        if self.num_index:
            G_out_num = tf.nn.sigmoid(G_logit[:, col_index:])
            G_out = tf.concat(values=[G_out, G_out_num], axis=1) if not empty_G_out else G_out_num
        return G_out

    # Discriminator
    @tf.function
    def discriminator(self, x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.leaky_relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        D_h2 = tf.nn.leaky_relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
        D_logit = tf.matmul(D_h2, self.D_W3) + self.D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    # loss function
    @tf.function
    def gain_Dloss(self, D_prob, mask):
        D_loss_temp = -tf.reduce_mean(mask * tf.math.log(D_prob + 1e-7) +
                                      (1 - mask) * tf.math.log(1. - D_prob + 1e-7))
        D_loss = D_loss_temp
        return D_loss

    @tf.function
    def gain_Gloss(self, sample, G_sample, D_prob, mask, n_classes):
        G_loss_temp = -tf.reduce_mean((1 - mask) * tf.math.log(D_prob + 1e-7))
        reconstruct_loss = 0

        # categorical loss
        current_ind = 0
        if self.cat_index:
            for j in range(len(n_classes)):
                M_current = mask[:, current_ind:current_ind + n_classes[j]]
                G_sample_temp = G_sample[:, current_ind:current_ind + n_classes[j]]
                X_temp = sample[:, current_ind:current_ind + n_classes[j]]
                reconstruct_loss += -tf.reduce_mean(
                    M_current * X_temp * tf.math.log(M_current * G_sample_temp + 1e-7)) / tf.reduce_mean(
                    M_current)
                current_ind += n_classes[j]
        # numerical loss
        if self.num_index:
            M_current = mask[:, current_ind:]
            G_sample_temp = G_sample[:, current_ind:]
            X_temp = sample[:, current_ind:]
            reconstruct_loss += tf.reduce_mean((M_current * X_temp - M_current * G_sample_temp) ** 2) / tf.reduce_mean(
                M_current)
        return G_loss_temp, reconstruct_loss

    # optimizer
    @tf.function
    def optimize_step(self, X_mb, M_mb, H_mb, n_classes):
        with tf.GradientTape() as g:
            # Generator
            G_sample = self.generator(X_mb, M_mb, n_classes)
            # Combine with observed data
            Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
            # Discriminator
            D_prob = self.discriminator(Hat_X, H_mb)
            D_loss = self.gain_Dloss(D_prob, M_mb)

        Dgradients = g.gradient(D_loss, self.theta_D)
        self.D_solver.apply_gradients(zip(Dgradients, self.theta_D))

        for i in range(3):
            with tf.GradientTape() as g:
                # Generator
                G_sample = self.generator(X_mb, M_mb, n_classes)
                # Combine with observed data
                Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
                # Discriminator
                D_prob = self.discriminator(Hat_X, H_mb)
                G_loss_temp, reconstructloss = self.gain_Gloss(X_mb, G_sample, D_prob, M_mb, n_classes)
                G_loss = G_loss_temp + self.alpha * reconstructloss
            Ggradients = g.gradient(G_loss, self.theta_G)
            self.G_solver.apply_gradients(zip(Ggradients, self.theta_G))
        return D_loss, G_loss_temp, reconstructloss

    def preprocess_data(self, data_x, data_m):
        data_train = np.array([])
        data_train_m = np.array([])
        # preprocess categorical variables
        if self.cat_index:
            data_cat = data_x[:, self.cat_index]
            data_cat_m = data_m[:, self.cat_index]
            data_cat_enc, data_cat_enc_miss = onehot_encoding(data_cat, data_cat_m, self.all_levels, has_miss=True)
            data_cat_enc = np.nan_to_num(data_cat_enc, 0)
            data_train = data_cat_enc
            data_train_m = data_cat_enc_miss
            n_classes = list(map(lambda x: len(x), self.all_levels))
        # preprocess numerical variables
        if self.num_index:
            data_num = data_x[:, self.num_index]
            data_num_m = data_m[:, self.num_index]
            data_num_norm, norm_parameters = normalization(data_num)
            data_num_norm = np.nan_to_num(data_num_norm, 0)
            data_train = np.concatenate([data_train, data_num_norm], axis=1) if data_train.size else data_num_norm
            data_train_m = np.concatenate([data_train_m, data_num_m], axis=1) if data_train_m.size else data_num_m

        num_encoded_cat = data_cat_enc.shape[1]
        num_num = data_num.shape[1]

        return data_train, data_train_m, n_classes, data_cat_enc_miss, norm_parameters, num_encoded_cat, num_num

    def __init__(self, num_hidden, cat_index, num_index, all_levels,
                 batch_size, hint_rate, alpha, iterations, num_imputations):
        self.cat_index = cat_index
        self.num_index = num_index
        self.all_levels = all_levels
        self.num_imputations = num_imputations
        self.num_hidden = num_hidden

        # System parameters
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations

        input_dim = self.num_hidden

        # Hidden state dimensions
        h_Gdim = int(input_dim)
        h_Ddim = int(input_dim)

        ## GAIN architecture
        # Discriminator variables
        self.D_W1 = tf.Variable(xavier_init([input_dim * 2, h_Ddim]))  # Data + Hint as inputs
        self.D_b1 = tf.Variable(tf.zeros(shape=[h_Ddim]))

        self.D_W2 = tf.Variable(xavier_init([h_Ddim, h_Ddim]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[h_Ddim]))

        self.D_W3 = tf.Variable(xavier_init([h_Ddim, input_dim]))
        self.D_b3 = tf.Variable(tf.zeros(shape=[input_dim]))  # Multi-variate outputs

        self.theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]

        # Generator variables
        # Data + Mask as inputs (Random noise is in missing components)
        self.G_W1 = tf.Variable(xavier_init([input_dim * 2, h_Gdim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[h_Gdim]))

        self.G_W2 = tf.Variable(xavier_init([h_Gdim, h_Gdim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[h_Gdim]))

        self.G_W3 = tf.Variable(xavier_init([h_Gdim, input_dim]))
        self.G_b3 = tf.Variable(tf.zeros(shape=[input_dim]))

        self.theta_G = [self.G_W1, self.G_W3, self.G_b1, self.G_b3]

        ## GAIN solver
        self.D_solver = tf.optimizers.Adam()
        self.G_solver = tf.optimizers.Adam()
        pass

    # let X be data_m, y be data_x
    def fit(self, X, y = None, X_test = None, y_test = None, check=False):
        data_x = X
        data_m = 1 - np.isnan(data_x).astype(np.float32)

        # preprocess
        data_train, data_train_m, n_classes, _, _, _, _ = self.preprocess_data(data_x, data_m)


        # Other parameters
        no, dim = data_x.shape
        input_dim = data_train.shape[1]

        # check model fitting
        if check:
            train_loss = []
            test_loss = []

        # Start Iterations
        Gloss_list = []
        Dloss_list = []
        pbar = tqdm(range(self.iterations))
        for i in pbar:
            # create mini batch
            indices = np.arange(no)
            np.random.shuffle(indices)
            for start_idx in range(0, no - self.batch_size + 1, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                X_mb = data_train[batch_idx, :]
                M_mb = data_train_m[batch_idx, :]

                # Sample random vectors
                Z_mb = uniform_sampler(0, 0.01, self.batch_size, input_dim)
                # Sample hint vectors
                H_mb_temp = binary_sampler(self.hint_rate, self.batch_size, input_dim)
                H_mb = M_mb * H_mb_temp

                # Combine random vectors with observed vectors
                X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

                D_loss_curr, G_loss_curr, reconstructloss = self.optimize_step(X_mb, M_mb, H_mb, n_classes)
                Gloss_list.append(G_loss_curr)
                Dloss_list.append(D_loss_curr)
                pbar.set_description(
                    "D_loss: {:.3f}, G_loss: {:.3f}, Reconstruction loss: {:.3f}".format(D_loss_curr.numpy(),
                                                                                         G_loss_curr.numpy(),
                                                                                         reconstructloss.numpy()))
                if check:
                    data_test_m = 1 - np.isnan(X_test).astype(np.float32)
                    y_pred = self.predict(X)
                    y_test_pred = self.predict(X_test)
                    train_loss.append(rmse_loss(y, y_pred, data_m))
                    test_loss.append(rmse_loss(y_test, y_test_pred, data_test_m))

        self.Gloss = G_loss_curr.numpy()
        self.Dloss = D_loss_curr.numpy()
        self.reconstructloss = reconstructloss.numpy()
        if check:
            return train_loss, test_loss
        pass

    def predict(self, X):
        data_x = X
        data_m = 1 - np.isnan(data_x).astype(np.float32)

        # preprocess
        data_train, data_train_m, n_classes, data_cat_enc_miss, norm_parameters, num_encoded_cat, num_num = self.preprocess_data(data_x, data_m)

        # Other parameters
        no, dim = data_x.shape
        input_dim = data_train.shape[1]

        # Return imputed data
        imputed_list = []
        for l in range(self.num_imputations):
            Z_mb = uniform_sampler(0, 0.01, no, input_dim)
            M_mb = data_train_m
            X_mb = data_train
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

            imputed_data = self.generator(X_mb, M_mb, n_classes)
            imputed_data = data_train_m * data_train + (1 - data_train_m) * imputed_data

            # revert onehot and renormalize
            imputed = np.empty(shape=(no, dim))
            if self.cat_index:
                imputed_cat = imputed_data[:, :num_encoded_cat]
                imputed_cat = onehot_decoding(imputed_cat, data_cat_enc_miss, self.all_levels, has_miss=False)
                imputed[:, self.cat_index] = imputed_cat
            if self.num_index:
                imputed_num = imputed_data[:, -num_num:]
                imputed_num = renormalization(imputed_num.numpy(), norm_parameters)
                imputed[:, self.num_index] = imputed_num
            imputed_list.append(imputed)
        return imputed_list[0]
    def score(self, X, y):
        # data_x_miss, data_x = X, y
        # imputed = self.predict(data_x_miss)
        # data_m = 1 - np.isnan(data_x_miss).astype(np.float32)
        # return rmse_loss(data_x, imputed, data_m)

        # loss = np.abs(self.Dloss - self.Gloss)
        # return loss
        return self.reconstructloss