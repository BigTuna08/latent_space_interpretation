import os
import numpy as np
import tensorflow as tf
from .models.vae_sequence import VAE

def save_datasets(loc, data, names):
    
    assert len(data) == len(names), "Each data needs 1 name!"
    
    data_loc = os.path.join(loc, "data")
    
    for l in [loc, data_loc]:
        try: os.makedirs(l)
        except: pass
        
    for name, d in zip(names, data):
        np.save(os.path.join(data_loc, name), d)
        
        
def load_datasets(loc):
    data_loc = os.path.join(loc, "data")

    datas = {}
    for data in os.listdir(data_loc):
        try:
            datas[data] = np.load(os.path.join(data_loc, data))
        except:
            print("failed to load", data, "(This is probably fine, unless it's needed)")
        
    return datas


def save_models(loc, models):
    
    model_loc = os.path.join(loc, "models")
    
    for l in [loc, model_loc]:
        try: os.makedirs(l)
        except: pass
    
    
    for m in models:
        m.save(os.path.join(model_loc, m.name))
      
        
def load_models(loc):

    model_loc = os.path.join(loc, "models")
    
    models = {}
    for model in os.listdir(model_loc):
        models[model] =  tf.keras.models.load_model(os.path.join(model_loc, model))
        

    return models
  
    
def save_ilvm(ilvm, loc):
    
    networks = ["recnet", "srn", "gm1", "gm2", "gm3"]
    settings = ["beta", "gamma", "kappa", "flow_depth"]
    
    settings_loc = loc + "/settings.txt"
    track_loc = loc + "/track.txt"
    
    
    try: os.mkdir(loc)
    except: pass
        
    
    models = [ilvm.__getattribute__(net) for net in networks]
    save_models(loc, models)
        
    
    with open(settings_loc, "w") as f:
        for s in settings:
            print("{}: {}".format(s, ilvm.__getattribute__(s)), file=f)
            
        
    with open(track_loc, "w") as f:
        for k, vs in ilvm.track.items():
            vals = "\t".join(map(str, vs))
            print("{}: {}".format(k, vals), file=f)



def save_seq_vae(vae, save_loc):
    save_models(save_loc, (vae.encoder, vae.decoder, vae.len_decoder))
    
    with open(save_loc + "/batch_size.txt", "w") as f:
        print(vae.batch_size, file=f)
        
    with open(save_loc + "/latent_dim.txt", "w") as f:
        print(vae.latent_dim, file=f)
        
    with open(save_loc + "/mean_seq_len.txt", "w") as f:
        print(vae.msl, file=f)
        
    with open(save_loc + "/start_word.txt", "w") as f:
        print(vae.start_word, file=f)
        
        
    with open(save_loc + "/hidden_units.txt", "w") as f:
        s = " ".join(map(str, vae.hidden_units))
        print(s, file=f)
        
    with open(save_loc + "/input_shape.txt", "w") as f:
        s = " ".join(map(str, vae.input_shape))
        print(s, file=f) 
        


def load_seq_vae(save_loc):

    all_models = load_models(save_loc)
    decoder = all_models["decoder"]
    len_decoder = all_models["len_decoder"]
    encoder = all_models["encoder"]

    with open(save_loc + "/batch_size.txt") as f:
        bs = int(f.read().strip())

    with open(save_loc + "/latent_dim.txt") as f:
        ld = int(f.read().strip())

    with open(save_loc + "/mean_seq_len.txt") as f:
        msl = float(f.read().strip())

    with open(save_loc + "/start_word.txt") as f:
        start = float(f.read().strip())


    with open(save_loc + "/hidden_units.txt") as f:
        hu = [int(x) for x in f.read().split()]

    with open(save_loc + "/input_shape.txt") as f:
        in_shape = [int(x) for x in f.read().split()]


    vae = VAE(inp_shape = in_shape, 
            latent_dim=ld,
            hidden_units=hu,
            batch_size=bs,
            optimizer=None,
            start_word=start)

    vae.encoder = encoder
    vae.decoder = decoder
    vae.len_decoder = len_decoder
    
    return vae