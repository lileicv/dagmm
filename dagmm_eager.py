'''
Deep Autoencoding Gaussian Mixture Model
'''
import time
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Sequential

tf.enable_eager_execution()

class DAGMM(tf.keras.Model):
    def __init__(self, xdim):
        '''
        Define the learnable parameters of the model
        K  -->  number of gaussian
        D  -->  dimension of z
        lambda1  -->  energy_loss weight
        lambda2  -->  sigma_diag_loss weight
        '''
        super(DAGMM, self).__init__(name='dagmm')

        D = 3
        K = 4
        
        self.encoder = Sequential([
            L.Dense(16, activation='tanh', input_shape=[xdim,]),
            L.Dense(8, activation='tanh'),
            L.Dense(1, activation='tanh')
        ])
        
        self.decoder = Sequential([
            L.Dense(8, activation='tanh', input_shape=[1,]),
            L.Dense(16, activation='tanh'),
            L.Dense(xdim)
        ])

        self.estimator = Sequential([
            L.Dense(8, activation='tanh', input_shape=[D,]),
            L.Dense(K, activation='softmax')
        ])

        self.phi   = tf.Variable(tf.zeros(shape=[K]),     dtype=tf.float32, name='phi')
        self.mu    = tf.Variable(tf.zeros(shape=[K,D]),   dtype=tf.float32, name='mu')
        self.sigma = tf.Variable(tf.zeros(shape=[K,D,D]), dtype=tf.float32, name='sigma')
        
        self.optimizer = tf.train.AdamOptimizer(0.001)
        self.step_counter = tf.train.get_or_create_global_step()

    def forward(self, x1, phase):
        '''
        input: x1
               phase ==> ['train', 'test', 'fix_gmm']
        return: z, x2, gamma
        '''
        z1 = self.encoder(x1)
        x2 = self.decoder(z1)
        # Concat Z
        dist_cos = tf.reduce_sum(x1*x2, axis=1, keepdims=True) / \
                (tf.norm(x1, axis=1, keepdims=True) * tf.norm(x2, axis=1, keepdims=True))
        dist_euc = tf.norm(x1-x2, axis=1, keepdims=True)
        z = tf.concat([z1, dist_cos, dist_euc], axis=1)
        gamma = self.estimator(z)
        phi, mu, sigma = self.calc_gmm(z, gamma)
        if phase=='train':
            energy = self.calc_energy(z, phi, mu, sigma)
        elif phase=='test':
            energy = self.calc_energy(z, self.phi, self.mu, self.sigma)
        elif phase=='fix_gmm':
            self.phi.assign(phi)
            self.mu.assign(mu)
            self.sigma.assign(sigma)
            return
        loss = self.loss(x1, x2, sigma, energy)
        return energy, loss

    def calc_gmm(self, z, gamma):
        # Calculate GMM param
        gamma_sum = tf.reduce_sum(gamma, axis=0)
        phi = tf.reduce_mean(gamma, axis=0)
        mu = tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:,None]
        z_centered = tf.sqrt(gamma[:,:,None]) * (z[:,None,:] - mu[None,:,:])
        sigma = tf.einsum(
                'ikl,ikm->klm', z_centered, z_centered) / gamma_sum[:,None,None]
        return phi, mu, sigma

    def calc_energy(self, z, phi, mu, sigma):
        # phi   K
        # mu    K,D
        # sigma K,D,D
        
        # 能量函数log phi项
        item1 = tf.expand_dims(tf.log(phi), axis=0) # 1,K
        # 分子项
        z_mu = z[:,None,:] - mu[None,:,:]           # N,K,D
        sigma_inv = tf.matrix_inverse(sigma)        # K,D,D
        item2 = -0.5*tf.reduce_sum( \
                tf.reduce_sum(tf.expand_dims(z_mu,axis=-1)*tf.expand_dims(sigma_inv,axis=0), axis=-2) * z_mu, \
                axis=-1) # N,K
        # 分母项
        sigma_det = tf.matrix_determinant(2*np.pi*sigma)
        item3 = tf.expand_dims(tf.log(tf.sqrt(sigma_det)), axis=0) # 1,K
        energy = - tf.reduce_sum(item1 + item2 - item3, axis=1)      # N
        return energy

    def loss(self, x1, x2, sigma, energy):
        lambda1 = 0.01
        lambda2 = 0.0001
        loss3 = lambda2 * tf.reduce_sum(tf.divide(1, tf.matrix_diag_part(sigma)))
        loss2 = lambda1 * tf.reduce_mean(energy)
        loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(x1-x2), axis=1), axis=0)
        loss = loss1 + loss2 + loss3
        return loss 

    def fit(self, x, batchsize, epochs, log_skip):
        ''' Fit on the training dataset
        '''
        print(epochs)
        x = tf.convert_to_tensor(x)
        for e in range(1, epochs+1):
            with tf.GradientTape() as tape:
                energy, loss = self.forward(x, 'train')
            grads = tape.gradient(loss, self.variables)
            self.optimizer.apply_gradients(zip(grads, self.variables), \
                                          global_step=self.step_counter)
            if e%log_skip == 0:
                print('epoch {}, loss {:.2f}'.format(e, loss))

        self.forward(x, 'fix_gmm')

    def predict(self, x):
        x = tf.convert_to_tensor(x)
        energy, _ = self.forward(x, 'test')
        return energy.numpy()

if __name__=='__main__':
    model = DAGMM(5)
