import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Lambda
from tensorflow.keras import backend as K
import numpy as np
import time
from .common_functions import sampling
#################################    Begn Toy Data      #######################################################

SCALE = 100
START = -1.0
loss_object = tf.keras.losses.MeanSquaredError(reduction='none')







def loss_function(real, pred, pred_lens):
        
    loss_ = loss_object(real, pred)
    
    mask = tf.squeeze(tf.sequence_mask(pred_lens, dtype=loss_.dtype, maxlen=real.shape[1]) )
    loss_ *= mask

    return loss_



def acc_metric(real, pred):
    
    pred = tf.round(pred*SCALE)
    real = tf.round(real*SCALE)

    loss_ = real != pred
    mask = tf.math.not_equal(real,0)
    loss_ = tf.logical_and(loss_,mask)


    missed = tf.reduce_sum(tf.cast(loss_, tf.float32))
    out_of = tf.reduce_sum(tf.cast(mask, dtype=tf.float32))

    return tf.stack((missed, out_of))


activation = None
class VAE:
    def __init__(self, inp_shape, 
                   latent_dim = 8,
                    hidden_units = [256, 32],
                    batch_size = 32,
                    optimizer= tf.keras.optimizers.Adam(),
                    start_word = "startsentance",
                     mean_seq_len = 16.
                ):
        

        self.msl = mean_seq_len
        
        enc_in = Input(shape=inp_shape, name="gru_in")
        output, state = GRU(hidden_units[0], 
                            return_state=True, 
                            activation=activation, 
                           )(enc_in)
        x = tf.concat([output,state], 1)
        for dim in hidden_units[1:]:
            x = Dense(dim, activation='relu')(x)

        ctx_mean = Dense(latent_dim, name='ctx_mean')(x)
        ctx_log_var = Dense(latent_dim, name='ctx_log_var')(x)
        ctx = Lambda(sampling, output_shape=(latent_dim,), name='ctx')([ctx_mean, ctx_log_var])   

        encoder = Model(enc_in, [ctx_mean, ctx_log_var, ctx], name="encoder")


        dec_in = Input(shape =(latent_dim, ))
        last_word_in = Input(shape=(1,))
        hidden_in = Input(shape=(hidden_units[0], ))

        x = tf.concat([tf.expand_dims(dec_in, 1), tf.expand_dims(last_word_in, 1)], axis=-1)
        for dim in reversed(hidden_units[1:]):
            x = Dense(dim, activation='relu')(x)
        gru_out, gru_state = GRU(hidden_units[0], 
                                 return_state=True, 
                                 return_sequences=True, 
                                 activation=activation, 
                                )(x, initial_state=hidden_in)
        
        
        dec_out = tf.reshape(gru_out, (-1, gru_out.shape[2]))
        dec_out = Dense(1)(dec_out)

        decoder = Model([last_word_in, dec_in, hidden_in], [dec_out, gru_state], name="decoder")   

        #######
        dec_len_in = Input(shape =(latent_dim, ))
        # x = dec_in
        x = dec_len_in 
        for dim in reversed(hidden_units[1:]):
            x = Dense(dim, activation='relu')(x)
        
        dec_len_out = Dense(1, 
                            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-2.0, maxval=4.0),
                           bias_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=10.0),)(x)

        decoder_len = Model(dec_len_in, dec_len_out, name="len_decoder")           
        #####
        
        self.len_decoder = decoder_len

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units

        self.encoder = encoder
        self.decoder = decoder

        self.start_word = start_word
        self.optimizer=optimizer
        self.input_shape = inp_shape



        self.track = {"acc_tr":[],
                        "acc_cv": [],
                        "loss_tr": [],
                        "loss_cv": []}
        


    def initialize_hidden_state(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        return tf.zeros((batch_size, self.hidden_units[0]))
        
        
    @tf.function
    def train_step(self, inp, true_lens):
        start_trace = time.time()

        recon_loss = 0.0

        acc = tf.constant([0,0], dtype=tf.float32)
        with tf.GradientTape() as tape:

            masked_inp = inp*tf.expand_dims(tf.sequence_mask(true_lens, dtype=tf.float32, maxlen=inp.shape[1]),-1)
            ctx_mean, ctx_log_var, context = self.encoder(masked_inp)
            
            kl_loss = 1 + ctx_log_var - K.square(ctx_mean) - K.exp(ctx_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            kl_loss = K.mean(kl_loss)
            
            pred_len = self.len_decoder(context)
            len_loss = tf.reduce_mean(loss_object(true_lens, pred_len))


            dec_input = tf.expand_dims([self.start_word] * self.batch_size, 1)
            dec_hidden = self.initialize_hidden_state()

            # Teacher forcing - feeding the target as the next input
            for t in range(inp.shape[1]):
                # passing enc_output to the self.decoder
                
                predictions, dec_hidden, = self.decoder((dec_input, context, dec_hidden))
                
                recon_loss += loss_function(inp[:, t], predictions , pred_len)
                acc += acc_metric(inp[:, t], predictions)
                
                # using teacher forcing
                dec_input = inp[:, t]
                
            recons_loss = tf.reduce_mean(tf.divide(recon_loss,pred_len))*self.msl
            loss = recons_loss + len_loss + kl_loss


        batch_loss = (loss / int(inp.shape[1])) 

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables + self.len_decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        
        trace_time = time.time() - start_trace
        print('Time taken for trace (train):\t\t{:.2f}\n'.format(trace_time))
        
        
        ####
        acc_m = 1 - acc[0]/acc[1]
        kl_m = (kl_loss / int(inp.shape[1])) 
        len_m = (len_loss / int(inp.shape[1])) 
        recons_m = (recons_loss / int(inp.shape[1])) 
        
        metrics = [(name, x )for x, name in zip([acc_m, kl_m, len_m, recons_m], ["acc_m", "kl_m", "len_m", "recons_m"])]

        return batch_loss, metrics

    
    @tf.function
    def train_step_tf(self, inp, true_lens):
        start_trace = time.time()

        recon_loss = 0.0

        acc = tf.constant([0,0], dtype=tf.float32)
        with tf.GradientTape() as tape:

            masked_inp = inp*tf.expand_dims(tf.sequence_mask(true_lens, dtype=tf.float32, maxlen=inp.shape[1]),-1)
            ctx_mean, ctx_log_var, context = self.encoder(masked_inp)
            
            kl_loss = 1 + ctx_log_var - K.square(ctx_mean) - K.exp(ctx_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            kl_loss = K.mean(kl_loss)
            
            pred_len = self.len_decoder(context)
            len_loss = tf.reduce_mean(loss_object(true_lens, pred_len))


            dec_input = tf.expand_dims([self.start_word] * self.batch_size, 1)
            dec_hidden = self.initialize_hidden_state()

            # No Teacher forcing 
            for t in range(inp.shape[1]):
                # passing enc_output to the self.decoder
                
                predictions, dec_hidden, = self.decoder((dec_input, context, dec_hidden))

                
                recon_loss += loss_function(inp[:, t], predictions , pred_len)
                acc += acc_metric(inp[:, t], predictions)
                

                # not using teacher forcing
                dec_input = predictions

                
            recons_loss = tf.reduce_mean(tf.divide(recon_loss,pred_len))*self.msl
            loss = recons_loss + len_loss + kl_loss


        batch_loss = (loss / int(inp.shape[1])) 

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables + self.len_decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        
        trace_time = time.time() - start_trace
        print('Time taken for trace (train):\t\t{:.2f}\n'.format(trace_time))
        
        
        ####
        acc_m = 1 - acc[0]/acc[1]
        kl_m = (kl_loss / int(inp.shape[1])) 
        len_m = (len_loss / int(inp.shape[1])) 
        recons_m = (recons_loss / int(inp.shape[1])) 
        
        metrics = [(name, x )for x, name in zip([acc_m, kl_m, len_m, recons_m], ["acc_m", "kl_m", "len_m", "recons_m"])]

        return batch_loss, metrics
    

    @tf.function
    def cv_step(self, inp, true_lens):
        start_trace = time.time()

        recon_loss = 0.0

        acc = tf.constant([0,0], dtype=tf.float32)
     

        masked_inp = inp*tf.expand_dims(tf.sequence_mask(true_lens, dtype=tf.float32, maxlen=inp.shape[1]),-1)
        ctx_mean, ctx_log_var, context = self.encoder(masked_inp)

        kl_loss = 1 + ctx_log_var - K.square(ctx_mean) - K.exp(ctx_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss = K.mean(kl_loss)

        pred_len = self.len_decoder(context)
        len_loss = tf.reduce_mean(loss_object(true_lens, pred_len))


        dec_input = tf.expand_dims([self.start_word] * self.batch_size, 1)
        dec_hidden = self.initialize_hidden_state()

        # Teacher forcing - feeding the target as the next input
        for t in range(inp.shape[1]):
            # passing enc_output to the self.decoder

            predictions, dec_hidden, = self.decoder((dec_input, context, dec_hidden))


            recon_loss += loss_function(inp[:, t], predictions , pred_len)
            acc += acc_metric(inp[:, t], predictions)


            # using teacher forcing
            dec_input = inp[:, t]

        recons_loss = tf.reduce_mean(tf.divide(recon_loss,pred_len))*self.msl
        loss = recons_loss + len_loss + kl_loss

        batch_loss = (loss / int(inp.shape[1])) 


        
        trace_time = time.time() - start_trace
        print('Time taken for trace (train):\t\t{:.2f}\n'.format(trace_time))
        
        
        ####
        acc_m = 1 - acc[0]/acc[1]
        kl_m = (kl_loss / int(inp.shape[1])) 
        len_m = (len_loss / int(inp.shape[1])) 
        recons_m = (recons_loss / int(inp.shape[1])) 
        
        metrics = [(name, x )for x, name in zip([acc_m, kl_m, len_m, recons_m], ["acc_m", "kl_m", "len_m", "recons_m"])]

        return batch_loss, metrics

#################################    End Toy Data      #######################################################