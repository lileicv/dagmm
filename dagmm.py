'''
Deep Autoencoding Gaussian Mixture Model
'''
import time
import numpy as np

import tensorflow as tf
from tensorflow.layers import Dense 

class DAGMM:
    def __init__(self, xdim):
        '''
        Define the learnable parameters of the model
        K  -->  number of gaussian
        D  -->  dimension of z
        lambda1  -->  energy_loss weight
        lambda2  -->  sigma_diag_loss weight
        '''
        lambda1 = 0.1
        lambda2 = 0.0001
        D = 3
        K = 4
        
        # Placeholder
        x1 = tf.placeholder(tf.float32, shape=[None, xdim])

        # Encoder
        x = Dense(16, activation='tanh', name='en-fc1')(x1)
        x = Dense(8, activation='tanh', name='en-fc2')(x)
        z1 = Dense(1, activation='tanh', name='en-fc3')(x)

        # Decoder
        x = Dense(8, activation='tanh', name='de-fc1')(z1)
        x = Dense(16, activation='tanh', name='de-fc2')(x)
        x2 = Dense(xdim, name='de-fc3')(x)

        # Concat Z
        dist_cos = tf.reduce_sum(x1*x2, axis=1, keep_dims=True) / \
                (tf.norm(x1, axis=1, keep_dims=True) * tf.norm(x2, axis=1, keep_dims=True))
        dist_euc = tf.norm(x1-x2, axis=1, keep_dims=True)
        z = tf.concat([z1, dist_cos, dist_euc], axis=1)
        
        # Estimation
        x = Dense(8, activation='tanh')(z)
        gamma = Dense(K, activation='softmax')(x) # K
        
        # Calculate GMM param
        gamma_sum = tf.reduce_sum(gamma, axis=0)
        phi = tf.reduce_mean(gamma, axis=0)
        mu = tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:,None]
        z_centered = tf.sqrt(gamma[:,:,None]) * (z[:,None,:] - mu[None,:,:])
        sigma = tf.einsum(
                'ikl,ikm->klm', z_centered, z_centered) / gamma_sum[:,None,None]

        phi2   = tf.Variable(tf.zeros(shape=[K]),     dtype=tf.float32, name='phi')
        mu2    = tf.Variable(tf.zeros(shape=[K,D]),   dtype=tf.float32, name='mu')
        sigma2 = tf.Variable(tf.zeros(shape=[K,D,D]), dtype=tf.float32, name='sigma')

        fix_gmm_op = tf.group(
            tf.assign(phi2,   phi),
            tf.assign(mu2,    mu),
            tf.assign(sigma2, sigma),
        )
        
        # Energy
        tr_energy = self.energy(z, [phi, mu, sigma])
        te_energy = self.energy(z, [phi2, mu2, sigma2])
       
        # Loss
        loss3 = lambda2 * tf.reduce_sum(tf.divide(1, tf.matrix_diag_part(sigma)))
        loss2 = lambda1 * tf.reduce_mean(tr_energy)
        loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(x1-x2), axis=1), axis=0)
        loss = loss1 + loss2 + loss3
        trainop = tf.train.AdamOptimizer(0.0001).minimize(loss)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        self.sess = sess
        self.x1 = x1
        self.x2 = x2
        self.tr_energy = tr_energy
        self.te_energy = te_energy
        self.loss = loss
        self.loss_list = [loss1, loss2, loss3]
        self.trainop = trainop
        self.fix_gmm_op = fix_gmm_op

    def energy(self, z, gmm_param):
        # phi   K
        # mu    K,D
        # sigma K,D,D
        phi, mu, sigma = gmm_param
        
        # 能量函数log phi项
        item1 = tf.expand_dims(tf.log(phi), axis=0) # 1,K
        # 分子项
        z_mu = z[:,None,:] - mu[None,:,:]           # N,K,D
        sigma_inv = tf.matrix_inverse(sigma)        # K,D,D
        item2 = -0.5*tf.reduce_sum(tf.reduce_sum(tf.expand_dims(z_mu,axis=-1)*tf.expand_dims(sigma_inv,axis=0), axis=-2) * z_mu, axis=-1) # N,K
        # 分母项
        sigma_det = tf.matrix_determinant(sigma_inv)
        item3 = tf.expand_dims(tf.log(tf.sqrt(np.pi*2*sigma_det)), axis=0) # 1,K
        energy = - tf.reduce_logsumexp(item1 + item2 - item3, axis=1)      # N
        return energy

    def energy2(self, z, gmm_param):
        '''
        The function of `energy2` is the same as `energy`
        '''
        phi, mu, sigma = gmm_param
        
        n_features = z.shape[1]
        min_vals = tf.diag(tf.ones(n_features, dtype=tf.float32)) * 1e-6
        L = tf.cholesky(sigma + min_vals[None,:,:])

        z_centered = z[:,None,:] - mu[None,:,:]  #ikl
        v = tf.matrix_triangular_solve(L, tf.transpose(z_centered, [1, 2, 0]))  # kli
        log_det_sigma = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), axis=1)
        d = z.get_shape().as_list()[1]
        logits = tf.log(phi[:,None]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                + d * tf.log(2.0 * np.pi) + log_det_sigma[:,None])
        energies = - tf.reduce_logsumexp(logits, axis=0)
        return energies
        
    def fit(self, x, batchsize, epochs, log_skip):
        ''' Fit on the training dataset
        '''
        for e in range(1, epochs+1):
            perm = np.random.permutation(x.shape[0])
            for batch in range(0, x.shape[0], batchsize):
                xbatch = x[perm[batch:batch+batchsize]]
                self.sess.run(self.trainop, feed_dict={self.x1:xbatch})
            
            if e%log_skip == 0:
                print(time.time())
                loss, loss_list= self.sess.run(
                    [self.loss, self.loss_list], feed_dict={self.x1:x})
                print('epoch {}, loss {:.2f} {:.2f} {:.2f} {:.2f}'.format( \
                    e, loss, loss_list[0], loss_list[1], loss_list[2]))
        # Fix the gmm param
        self.sess.run(self.fix_gmm_op, feed_dict={self.x1:x})

    def predict(self, x):
        # fix gmm op
        output = self.sess.run(self.te_energy, feed_dict={self.x1:x})
        return output

if __name__=='__main__':
    model = DAGMM(5)
