# Lens Model Experiments  
For each dataset, there are 3 notebooks. These are for creating datasets (dataset), creating and fitting the vae for the dataset (vae) and creating and fitting the lens models (lens)  

The digit_thick notebooks use the MNIST dataset, the digit_rotation notebooks use an altered version of MNIST which contains rotated images and the toy_sequence notebooks use a dataset containing sequences of numbers. For more information, please see the relavent notebooks ending in "dataset.ipynb"

To run the experiments for a given dataset, you must run the 3 notebooks in order. The first (ending in dataset) will set up the dataset. The second (ending in vae) creates and trains the VAE. The final (ending in lens) creates and trains the lens model. 

The remaining notebook, "normalizing_flows_density_estimation", runs a experiment using normalizing flows for density estimation.
