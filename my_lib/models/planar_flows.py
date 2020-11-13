import tensorflow as tf
import numpy as np
import keras
from keras import layers
from keras import backend as K

def get_batch_corrected_u(w, u):
    wu = tf.reduce_sum( tf.multiply( w, u), 1)
    m = -1.0 + tf.nn.softplus(wu)
    inner = tf.subtract(m, wu) # m(wu) - uw

    to_add =tf.transpose( inner * tf.transpose(w) / tf.math.sqrt(tf.reduce_sum(w ** 2.0)))

    return tf.add(u, to_add)  # uhat, ensures invertable



class PlanarFlow(tf.keras.layers.Layer):
    
    def __init__(self):
        super().__init__()

        
        
    def call(self, z, w, u, b):

        u = get_batch_corrected_u(w, u) # ensure invertable
        ######
        bs, d = z.shape
    
        b = tf.reshape(b, [bs])
        
        wTz = tf.reduce_sum( tf.multiply( w, z ), 1)

        inner = tf.add(wTz, b)
        to_add = tf.matmul(tf.linalg.diag(self.h(inner)), u)
        z = tf.add(z, to_add)

        ####
        log_det = tf.math.log(self.det_jcb(z, w, u, b))
            
        return z, log_det
    
    
    def det_jcb(self, z, w, u, b):
        bs, d = z.shape
        
        wTz = tf.reduce_sum( tf.multiply( w, z ), 1)
        inner = tf.add(wTz, b)

        psi = tf.matmul(tf.linalg.diag(self.h_prime(inner)), w)  # tf.tensordot(w, self.h_prime(inner), axes=0)
        
        to_add = tf.reduce_sum( tf.multiply( u , psi ), 1)
        
        inner = tf.add(tf.ones((bs)), to_add)
        return tf.abs(inner)
    
    
    def h(self, y):
        return tf.math.tanh(y)
    

    def h_prime(self, y):
        return 1.0 - tf.math.tanh(y) ** 2.0
    
        
    
class FixedPlanarFlow(tf.keras.layers.Layer):
    def __init__(self, w, uhat, b, 
                 store_alpha = True,
                 update_freq=None, 
                 optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.6),
                 store_iter = "all",  # options False/None, "last", "all"
                 name="fixed_pf", **kwargs):
        
        super().__init__(name=name, **kwargs)
        
        self.u = uhat
        self.w = w
        self.b = b
        
        self.store_alpha = store_alpha
        self.prev_alphas = []
        self.update_freq = update_freq
        self.optimizer = optimizer
        if store_iter:
            if store_iter in ["last", "all"]:
                self.store_iter = store_iter
                self.iter_count = [-1]
            else: raise Exception("store_iter must be: False/None, last, all")
        

        
    def call(self, z_prev):
        inner = tf.add(tf.tensordot(self.w, z_prev, axes=1), self.b)
        to_add = tf.tensordot(self.u, self.h(inner), axes=0)
        z_next = tf.add(z_prev, to_add)
        return z_next
    
    
    def inv_call(self, z_next, 
                 init=0.0, max_iter = 10000, tol=1e-12,):
        
        
        alpha = self.compute_alpha(z_next, init, max_iter, tol)
        
        z_par =  alpha * self.w / tf.reduce_sum(tf.square(self.w))
        
        inner = tf.add(tf.tensordot(self.w, z_par, axes=1), self.b)
        to_subt = tf.tensordot(self.u, self.h(inner), axes=0)

        z_prev = z_next  - to_subt

        return z_prev
        
    

    def compute_alpha(self, fz, init, max_iter, tol):
         
        if self.store_alpha and self.prev_alphas:
            init = self.prev_alphas[-1]
        alpha = tf.Variable(init)
        converged=False
        
        for i in range(max_iter):
            
            with tf.GradientTape() as tape:
                wTu = tf.tensordot(self.w, self.u, axes=1)
                wTfz= tf.tensordot(self.w, fz, axes=1)

                diff = alpha + wTu*self.h(alpha + self.b) - wTfz
                loss = diff **2
                
            variables = [alpha]
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))


            if self.update_freq:
                if i % self.update_freq == 1:
                    print("after", i, "iters, alpha=", alpha, ", loss=", loss)


            if tf.reduce_max(loss) < tol:
                converged = True
                if self.update_freq:
                    print("Converged after", i, "iters")
                
                
            if converged:

                if self.store_iter:
                    if self.store_iter == "all":
                        self.iter_count.append(i)
                    else:
                        self.iter_count[0] = i
                        
                if self.store_alpha:
                    self.prev_alphas.append(alpha)

                return alpha
        
        
        # loop finished w/o converging
        print("Failed to converge")
        print("after", i, "iters, alpha=", alpha, ", loss=", loss)
        raise Exception("Did not converge!")
            
            
    
    
    def det_jcb(self, z):
        psi = self.psi(z)
        inner = tf.add(1, tf.tensordot(self.u, psi, axes=1))
        return tf.abs(inner)
    
    
    def psi(self, z):
        inner = tf.add(tf.tensordot(self.w, z, axes=1), self.b)
        psi_z = tf.tensordot(self.w, self.h_prime(inner), axes=0)
        return psi_z
    
    def h(self, y):
        return tf.math.tanh(y)
    
    def h_prime(self, y):
        return 1.0 - tf.math.tanh(y) ** 2.0
    
    def m(self, x):
        return -1.0 + tf.nn.softplus(x)

            
  
def create_inv_flow_layers(w_, u_, b_, samples=[0]):
    flow_len = K.int_shape(w_)[-1]
    
    uhat = np.zeros((u_.shape))
    for i in range(flow_len):
        uhat[:,:,i] = get_batch_corrected_u(w_[:,:, i], u_[:,:,i])
    uhat = tf.constant(uhat, dtype=tf.float32)
    
    flow_chains = []
    
    for si in samples:
        flow_chain = []
        for layer_i in range(flow_len):
            flow = FixedPlanarFlow(w_[si,:,layer_i], uhat[si,:,layer_i], b_[si,:,layer_i], name="fixed_pf_{}".format(layer_i))
            flow_chain.append(flow)
        flow_chains.append(flow_chain)
        
    return flow_chains