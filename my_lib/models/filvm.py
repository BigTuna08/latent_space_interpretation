import tensorflow as tf
import os
import time
from tensorflow.keras.losses import MSE, categorical_crossentropy
import numpy as np

class FILVM:
    def __init__(self, rn, srn, pf_layers, gm, optimizer, beta=1.0, gamma=1.0):
        self.recnet = rn
        self.srn = srn
        self.pf_layers = pf_layers
        self.gm1, self.gm2, self.gm3 = gm
        
        self.optimizer = optimizer
        self.beta = beta
        self.gamma = gamma
        
        self.track = {}
        self.ld = self.gm1.input_shape[1]
        self.flow_depth = len(pf_layers)
        
    
    def set_metrics(self, metrics):
        self.metrics = metrics
        
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
        
        
    def update_track(self, batch_loss, met_res, pref, verb=1):

        key = "{}_full_loss".format(pref)
        v = batch_loss.numpy()
        self._update_track(key, v)

        for name, val in  met_res:
            try:
                key = "{}_{}".format(pref, name.numpy().decode('UTF-8'))
            except:
                key = "{}_{}".format(pref, name)
            v = val.numpy()
            self._update_track(key, v)

            if verb:
                print(key, v)     
                
    
        
    def _update_track(self, k, v):
        hist = self.track.get(k, [])
        hist.append(v)
        self.track[k] = hist
    
    
    def fit(self, epochs, log_dir, ds1, ds2, ds_test, batch_size, plot_freq = 1, save=True, show=True, early_stop=None, watch=None):
        try:
            os.mkdir(log_dir)
        except: pass
        

        steps_per_epoch = len(ds1)//batch_size
        steps_per_epoch_cv = len(ds_test)//batch_size


        full_start = time.time()
        gotnan = False
        for epoch in range(epochs):
            start = time.time()

            total_loss1, total_loss2, cv_total_loss = 0, 0, 0

            for (batch, (inp1, inp2)) in enumerate(zip(ds1, ds2)):

                batch_loss1, met_res1 = self.train_step1(inp1)
                total_loss1 += batch_loss1

                batch_loss2, met_res2 = self.train_step2(inp2)
                total_loss2 += batch_loss2


                
                if np.isnan(batch_loss1.numpy()):
                    print("Found nan!!!")
                    gotnan = True

                #2500
                if batch % 2500 == 0 or gotnan:
                    print('Epoch {} Batch {} Loss 1 {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss1.numpy()))


                    self.update_track(batch_loss1, met_res1,  pref="tr1")
                    print("\n")



                    print('Epoch {} Batch {} Loss 2 {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss2.numpy()))

                    
                    
                    self.update_track(batch_loss2, met_res2, pref="tr2")
                    print("\n")

                if gotnan: break


            print('Epoch {} Loss {:.4f}'.format(epoch + 1, (total_loss1+total_loss2) / steps_per_epoch) )


            for (batch, inp) in enumerate(ds_test):

                batch_loss, met_res = self.cv_step1(inp)
                cv_total_loss += batch_loss  


            print("*"*10, "CV", "*"*10)      
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, cv_total_loss / steps_per_epoch_cv))

            self.update_track(batch_loss, met_res, pref="cv")

    
            if early_stop and epoch > early_stop:
                if min(self.track[watch]) < min(self.track[watch][-early_stop:]):
                    break
            if gotnan: break
            

        print('\n\nFull Time taken for {} epochs is {} sec\n\n'.format(epochs, time.time() - full_start))
        

        
    def in_to_zt(self, inp):
        loss = 0


        # if inp is correct, one will fail and correct one will succeed 
        try:
            z0_, log_q_z0, w_, u_, b_ = self.recnet(inp)[2:]
        except Exception as e1:
            try:
                pred_s = self.srn(inp)
                z0_, log_q_z0, w_, u_, b_, = self.recnet((inp, *pred_s))[2:]

            except Exception as e2:
                print("Error in in_to_zt!!")
                print("e1:", e1)
                print("e2:", e2)
                raise Exception("See prints!")
                

        w_ = tf.reshape(w_, (-1, self.ld, self.flow_depth))
        u_ = tf.reshape(u_, (-1, self.ld, self.flow_depth))
        b_ = tf.reshape(b_, (-1, 1, self.flow_depth))
        z_ = [z0_]

        log_dets = []

        for i in range(self.flow_depth):
            z_i, log_det_i = self.pf_layers[i](z_[-1], w_[:, :, i], u_[:, :, i], b_[:, :, i])

            z_.append(z_i)
            log_dets.append(log_det_i)
            
        return z_[-1]
        
        
    @tf.function
    def train_step1(self, inp):
        loss = 0

        with tf.GradientTape() as tape:

            input_z, input_s_d, input_s_t = inp

            z0_, log_q_z0, w_, u_, b_ = self.recnet(inp)[2:]

            w_ = tf.reshape(w_, (-1, self.ld, self.flow_depth))
            u_ = tf.reshape(u_, (-1, self.ld, self.flow_depth))
            b_ = tf.reshape(b_, (-1, 1, self.flow_depth))
            z_ = [z0_]

            log_dets = []

            for i in range(self.flow_depth):
                z_i, log_det_i = self.pf_layers[i](z_[-1], w_[:, :, i], u_[:, :, i], b_[:, :, i])

                z_.append(z_i)
                log_dets.append(log_det_i)

#             z_pred, *s_pred = self.gm(z_[-1])
            z_pred = self.gm1(z_[-1])
            s_pred = self.gm2(z_[-1]), self.gm3(z_[-1])

            ##  Rename ##

            s = (input_s_d, input_s_t) 
            log_dets = tf.reduce_sum(log_dets, axis=0) # sum over flow len

            args = z_pred, input_z, s_pred, s, z0_, log_q_z0, log_dets, z_[-1]

            ##  Compute loss / self.metrics ##
            loss = self.loss_fn(*args, beta=self.beta, gamma=self.gamma)
            metric_res = [(name, fn(*args)) for name, fn in self.metrics] 


        variables = self.recnet.trainable_variables + self.gm1.trainable_variables+ self.gm2.trainable_variables+ self.gm3.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, metric_res


    @tf.function
    def cv_step1(self, inp):

        loss = 0 

        input_z, input_s_d, input_s_t = inp
        z0_, log_q_z0, w_, u_, b_ = self.recnet(inp)[2:]


        w_ = tf.reshape(w_, (-1, self.ld, self.flow_depth))
        u_ = tf.reshape(u_, (-1, self.ld, self.flow_depth))
        b_ = tf.reshape(b_, (-1, 1, self.flow_depth))
        z_ = [z0_]

        log_dets = []

        for i in range(self.flow_depth):
            z_i, log_det_i = self.pf_layers[i](z_[-1], w_[:, :, i], u_[:, :, i], b_[:, :, i])

            z_.append(z_i)
            log_dets.append(log_det_i)

        z_pred = self.gm1(z_[-1])
        s_pred = self.gm2(z_[-1]), self.gm3(z_[-1])

        ##  Rename ##

        s = (input_s_d, input_s_t) 
        log_dets = tf.reduce_sum(log_dets, axis=0) # sum over flow len

        args = z_pred, input_z, s_pred, s, z0_, log_q_z0, log_dets, z_[-1]

        ##  Compute loss / self.metrics ##
        loss = self.loss_fn(*args, beta=self.beta, gamma=self.gamma)
        metric_res = [(name, fn(*args)) for name, fn in self.metrics] 

        return loss, metric_res


    ################

    @tf.function
    def train_step2(self, inp):
        loss = 0
        loss_srn = 0 
        
        input_z, input_s_d, input_s_t = inp
        
        with tf.GradientTape() as tape1:
    
            pred_s_d, pred_s_t = self.srn(input_z)
            
#             loss_srn_d = tf.reduce_mean(categorical_crossentropy(input_s_d, pred_s_d,) )
            loss_srn_d = tf.reduce_mean(MSE(input_s_d, pred_s_d,) )
            loss_srn_t = tf.reduce_mean(MSE(input_s_t, pred_s_t,) )
            loss_srn = loss_srn_d + loss_srn_t
            
            
        variables = self.srn.trainable_variables 
        gradients = tape1.gradient(loss_srn, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        
        with tf.GradientTape() as tape2:

            z0_, log_q_z0, w_, u_, b_ = self.recnet((input_z, pred_s_d, pred_s_t))[2:]


            w_ = tf.reshape(w_, (-1, self.ld, self.flow_depth))
            u_ = tf.reshape(u_, (-1, self.ld, self.flow_depth))
            b_ = tf.reshape(b_, (-1, 1, self.flow_depth))
            z_ = [z0_]

            log_dets = []

            for i in range(self.flow_depth):
                z_i, log_det_i = self.pf_layers[i](z_[-1], w_[:, :, i], u_[:, :, i], b_[:, :, i])

                z_.append(z_i)
                log_dets.append(log_det_i)

            z_pred = self.gm1(z_[-1])
            s_pred = self.gm2(z_[-1]), self.gm3(z_[-1])

            ##  Rename ##

#             s = (pred_s_d, pred_s_t)  # predicted s from rec net
            s = (input_s_d, input_s_t)  # cheat and use labels
            
            log_dets = tf.reduce_sum(log_dets, axis=0) # sum over flow len

            args = z_pred, input_z, s_pred, s, z0_, log_q_z0, log_dets, z_[-1]

            ##  Compute loss / self.metrics ##
            loss = self.loss_fn(*args, beta=self.beta, gamma=self.gamma)
            metric_res = [(name, fn(*args)) for name, fn in self.metrics] 
            
        metric_res += [("loss_srn_d", loss_srn_d), ("loss_srn_t", loss_srn_t), ("loss_srn", loss_srn)]

        variables = self.recnet.trainable_variables + self.gm1.trainable_variables+ self.gm2.trainable_variables+ self.gm3.trainable_variables
        gradients = tape2.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, metric_res


    @tf.function
    def cv_step2(self, inp):

        loss = 0 

        input_z, input_s_d, input_s_t = inp

        pred_s_d, pred_s_t = self.srn(input_z)
        z0_, log_q_z0, w_, u_, b_ = self.recnet((input_z, pred_s_d, pred_s_t))[2:]


        w_ = tf.reshape(w_, (-1, self.ld, self.flow_depth))
        u_ = tf.reshape(u_, (-1, self.ld, self.flow_depth))
        b_ = tf.reshape(b_, (-1, 1, self.flow_depth))
        z_ = [z0_]

        log_dets = []

        for i in range(self.flow_depth):
            z_i, log_det_i = self.pf_layers[i](z_[-1], w_[:, :, i], u_[:, :, i], b_[:, :, i])

            z_.append(z_i)
            log_dets.append(log_det_i)

        z_pred = self.gm1(z_[-1])
        s_pred = self.gm2(z_[-1]), self.gm3(z_[-1])

        ##  Rename ##

        s = (pred_s_d, pred_s_t)  # predicted s from rec net
        log_dets = tf.reduce_sum(log_dets, axis=0) # sum over flow len

        args = z_pred, input_z, s_pred, s, z0_, log_q_z0, log_dets, z_[-1]

        ##  Compute loss / self.metrics ##
        loss = self.loss_fn(*args, beta=self.beta, gamma=self.gamma)
        metric_res = [(name, fn(*args)) for name, fn in self.metrics] 

        return loss, metric_res   
    
    
    