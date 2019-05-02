'''
Deep Autoencoding Gaussian Mixture Model
'''

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import Sequential

class DAGMM:
    def __init__(self, xdim):
        '''
        Define the learnable parameters of the model
        K  -->  number of gaussian
        D  -->  dimension of z
        lambda1  -->  energy_loss weight
        lambda2  -->  sigma_diag_loss weight
        '''
        self.lambda1 = 0.1
        self.lambda2 = 0.0001
        self.sess = tf.Session()
        self.D = D = 3
        self.K = K = 4
        self.xdim = xdim

        self.encoder = Sequential([
            Dense(16, activation='tanh', input_shape=(xdim,)),
            Dense(8,  activation='tanh'),
            Dense(1)])

        self.decoder = Sequential([
            Dense(8,  activation='tanh', input_shape=(1,)),
            Dense(16, activation='tanh'),
            Dense(xdim)
        ])

        self.estimation = Sequential([
            Dense(8, activation='tanh', input_shape=(D,)),
            Dense(K, activation='softmax')
        ])
        
        # GMM param for test
        self.phi   = tf.Variable(tf.zeros(shape=[K]),     dtype=tf.float32, name='phi')
        self.mu    = tf.Variable(tf.zeros(shape=[K,D]),   dtype=tf.float32, name='mu')
        self.sigma = tf.Variable(tf.zeros(shape=[K,D,D]), dtype=tf.float32, name='sigma')
        self.L     = tf.Variable(tf.zeros(shape=[K,D,D]), dtype=tf.float32, name='L')
       
        self.forward()

    def forward(self):
        ''' Build Graph
        '''
        tf.set_random_seed(123)

        x = tf.placeholder(dtype=tf.float32, shape=[None, self.xdim])
        z_c = self.encoder(x)
        x2 = self.decoder(z_c)
        z = self.generate_z(x, x2, z_c)
        gamma = self.estimation(z)
        phi, mu, sigma, L = self.estimate_gmm(gamma, z)
        tr_energy = self.energy(z, [phi, mu, sigma, L])
        te_energy = self.energy(z, None)

        fix_gmm_op = tf.group(
            tf.assign(self.phi, phi),
            tf.assign(self.mu, mu),
            tf.assign(self.sigma, sigma),
            tf.assign(self.L, L)
        )
        loss3 = self.lambda2 * tf.reduce_sum(tf.divide(1, tf.matrix_diag_part(sigma)))
        loss2 = self.lambda1 * tf.reduce_mean(tr_energy)
        loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(x-x2), axis=1), axis=0)
        loss = loss1 + loss2 + loss3
        trainop = tf.train.AdamOptimizer(0.0001).minimize(loss)
        self.sess.run(tf.global_variables_initializer())
 
        self.x = x
        self.x2 = x2
        self.tr_energy = tr_energy
        self.te_energy = te_energy
        self.loss = loss
        self.loss1 = loss1
        self.loss2 = loss2
        self.loss3 = loss3
        self.trainop = trainop
        self.fix_gmm_op = fix_gmm_op

    def energy(self, z, gmm_param):
        '''
        Instead of inverse covariance matrix, exploit cholesky decomposition
        for stability of calculation.
        '''
        if gmm_param is not None:
            phi, mu, sigma, L = gmm_param
        else:
            mu      = self.mu
            phi     = self.phi
            sigma   = self.sigma
            L       = self.L
        z_centered = z[:,None,:] - mu[None,:,:]  #ikl
        v = tf.matrix_triangular_solve(L, tf.transpose(z_centered, [1, 2, 0]))  # kli
        log_det_sigma = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), axis=1)
        d = z.get_shape().as_list()[1]
        logits = tf.log(phi[:,None]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                + d * tf.log(2.0 * np.pi) + log_det_sigma[:,None])
        energies = - tf.reduce_logsumexp(logits, axis=0)
        return energies

    def estimate_gmm(self, gamma, z):
        ''' Estimate GMM param
        '''
        gamma_sum = tf.reduce_sum(gamma, axis=0)
        phi = tf.reduce_mean(gamma, axis=0)
        mu = tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:,None]
        z_centered = tf.sqrt(gamma[:,:,None]) * (z[:,None,:] - mu[None,:,:])
        sigma = tf.einsum(
                'ikl,ikm->klm', z_centered, z_centered) / gamma_sum[:,None,None]
        # Calculate a cholesky decomposition of covariance in advance
        n_features = z.shape[1]
        min_vals = tf.diag(tf.ones(n_features, dtype=tf.float32)) * 1e-6
        L = tf.cholesky(sigma + min_vals[None,:,:])
        return phi, mu, sigma, L
    
    def generate_z(self, x, x2, z_c):
        '''
        Based on the original paper, features of reconstraction error
        and zc are composed.

        Input
            x   : src sample
            x2  : reconstruct sample
            z_c : encoding vector
        Output
            z = [Euclidean distance, cosine similarity, z_c]
        '''

        def euclid_norm(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
        
        norm_x = euclid_norm(x)
        norm_x2 = euclid_norm(x2)
        dist_x = euclid_norm(x - x2)
        dot_x = tf.reduce_sum(x * x2, axis=1)
        
        min_val = 1e-3
        loss_E = dist_x  / (norm_x + min_val)
        loss_C = 0.5 * (1.0 - dot_x / (norm_x * norm_x2 + min_val))
        return tf.concat([z_c, loss_E[:,None], loss_C[:,None]], axis=1)
    
    def fit(self, x, batchsize, epochs, log_skip):
        ''' Fit on the training dataset
        '''
        for e in range(1, epochs+1):
            perm = np.random.permutation(x.shape[0])
            for batch in range(0, x.shape[0], batchsize):
                xbatch = x[perm[batch:batch+batchsize]]
                self.sess.run(self.trainop, feed_dict={self.x:xbatch})
            
            if e%log_skip == 0:
                loss, loss1, loss2, loss3, x2 = self.sess.run(
                    [self.loss, self.loss1, self.loss2, self.loss3, self.x2], feed_dict={self.x:x})
                print('epoch {}, loss {:.2f} {:.2f} {:.2f} {:.2f}'.format(e, loss, loss1, loss2, loss3))
        # Fix the gmm param
        self.sess.run(self.fix_gmm_op, feed_dict={self.x:x})

    def predict(self, x):
        # fix gmm op
        output = self.sess.run(self.te_energy, feed_dict={self.x:x})
        return output


