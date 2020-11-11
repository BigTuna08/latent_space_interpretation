import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def make_id(params):
    dep = "dep-{}".format(params["flow_depth"])
    inner_fmt = "inr-" + "{}_" * len(params["inter_dims"])
    inner = inner_fmt[:-1].format(*params["inter_dims"])
    opt = "opt-{}".format(params["optimizer"].get_config()["name"])
    b = "b-{}".format(params["beta"]).replace(".", "_")
    g = "g-{}".format(params["gamma"]).replace(".", "_") 
    return "--".join([dep, inner, opt, b, g])


def param_search_to_list(full_params, parts={}):
    
    full_params = full_params.copy() # don't destroy original

    l = []
    k, vs = full_params.popitem()

    for v in vs:
        d = parts.copy()
        d[k] = v
        
        if len(full_params) < 1:
            l.append(d)
        else:
            l += param_search_to_list(full_params, d)
            
    return l

###########

def make_plots(inp_data, ilvm, save_loc, epoch, prefixs="", cs=None, show=True, save=True, return_zt=False):
    
    if not show and not save: return
    
    if type(prefixs) != list:
        prefixs = [prefixs]
        cs = [cs]
    
    z_t = ilvm.in_to_zt(inp_data)
    
    assert len(prefixs) == len(cs), "Prefixs and cs must be same len!"
    
    if save:
        try: os.mkdir(save_loc)
        except: pass
    
    for pref, c in zip(prefixs, cs):
            
        plot_z(z_t, c, "Color by {} after {} epochs".format(pref, epoch))

        if save:
            try: os.mkdir("{}/{}".format(save_loc, pref))
            except: pass

            plt.savefig("{}/{}/epoch_{}.png".format(save_loc, pref, epoch))

        if show:
            plt.show()

        plt.clf()
        
    if return_zt:
        return z_t

    
    
def plot_z(z, c=None, title="", zname="z*", dims=[0,1], alpha=0.5):
    d0, d1 = dims
    plt.scatter(z[:, d0], z[:, d1], c=c, alpha=alpha)
    plt.xlabel("{}[{}]".format(zname, d0))
    plt.ylabel("{}[{}]".format(zname, d1))
    plt.title(title)
    plt.colorbar()
    
    
import tensorflow_probability as tfp
def walk_ld(model,
            decoder,
            xrng= (0.025, 0.975),
            yrng = (-3, 3),
            x_pspace = True,
            y_pspace = False,
            dims = [0,1],
            n=30,
            figsize=(10, 10),
            model_name="vae_mnist",
            digit_size = 28):
    
    
    filename = os.path.join(model_name, "digits_over_latent.png")
    figure = np.zeros((digit_size * n, digit_size * n))
    norm = tfp.distributions.Normal(0, 1)
    z_dim = model.gm1.input_shape[1]
    
    
    if x_pspace:
        grid_x = norm.quantile(np.linspace(*xrng, n))
    else:
        grid_x = np.linspace(*xrng, n)
        
    if y_pspace:
        grid_y = norm.quantile(np.linspace(*yrng, n))[::-1]
    else:
        grid_y = np.linspace(*yrng, n)[::-1]
    
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            
    
            z_t = tf.reshape(np.array([xi,yi] + [0]*(z_dim-2)), (1,z_dim))
            z_pred = model.gm1(z_t)
            digit = decoder(z_pred).numpy().reshape((28,28))
            
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

            
    plt.figure(figsize=figsize)
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[{}]".format(dims[0]))
    plt.ylabel("z[{}]".format(dims[1]))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()
    
########
