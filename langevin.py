from utils import *
import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import os
import sys



# --> set the correct path to save simulation results
path = "/home/clod/Desktop/Thesis/toy-model/langevin/"

metropolis = False #True # set to True to use metropolis sampling 

if metropolis:
    path = "/home/clod/Desktop/Thesis/toy-model/langevin_metropolis/"


potential_fn = mueller_brown 
#potential_fn = gaussian_potential 


# set to True to use multiple cores
use_multiple_cores = True
n_cores = 20 # set to None to use all cores, skip is the previous line is set to False
            # NOTE: for KDE, I use all the possible cores anyway

#### Starting parameters ####
#T = 20
beta = 0.03  #1/T 
gamma = 0.5
dt = 5e-4

num_pts = 700000# number of points for each Langevin dynamics
M = 1 #number of starting points for Langevin dynamics ( ignore this if use_multiple_cores is True)

x1_min, x1_max =-2.0, 1.0
x2_min, x2_max = -0.5, 2.0

gridsize = 100 # number of points for the hexbin plot

###########################################################################################################

if metropolis:
    if potential_fn == gaussian_potential:
        path += "gaussian_potential/"
    elif potential_fn == mueller_brown:
        path += "mueller_brown/"


if use_multiple_cores:
    # set the number of cores to use
    if n_cores is None:
        n_cores = min(M, os.cpu_count() )
       
    

print("---------------------")
print("Starting simulation with parameters:")
print("beta = ", beta)
print("gamma = ", gamma)
print("dt = ", dt)
print("num_pts = ", num_pts)
print("M = ", M)    
print("---------------------")

# Set the random seed for reproducibility
#np.random.seed(42)
#torch.manual_seed(42)

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    # Check if the directory exists
    if not os.path.exists(path):
        sys.exit(f"Directory {path} does not exist.")
        
    dir_name = "beta={}_gamma={}_dt={}/nmpts={}/".format(beta, gamma, dt, num_pts*M)
    folder_path = os.path.join(path, dir_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory {folder_path} created.")
        
    # plot the potential
    plot_potential(potential_fn, x1_min, x1_max, x2_min, x2_max, save_path=os.path.join(folder_path+"mueller_brown_potential.png"))

    # starting sequence of point to start Langevin dynamics
    starting_sequence = np.random.uniform(low=[x1_min, x2_min], high=[x1_max, x2_max], size=(M, 2))


    if not use_multiple_cores:
        
        # for this function W_list was to check that the kicks follow the normal distribution
        # ans starting_sequence saves the starting point in the Langevin dynamics
        if metropolis:

            data, start_seq, W_list, Pacc = Od_Langevin(beta,gamma,dt, potential_fn, grad_U_torch,
                                                num_points = num_pts,starting_sequence=starting_sequence,
                                                metropolis=metropolis)
        else:   
            data, start_seq, W_list = Od_Langevin(beta,gamma,dt, potential_fn, grad_U_torch, 
                                                num_points = num_pts,starting_sequence=starting_sequence,
                                                metropolis=metropolis)
    else:
        if metropolis:

            data, start_seq, W_list, Pacc = parallel_Od_Langevin(beta,gamma,dt, potential_fn, grad_U_torch,
                                                        num_points = num_pts,
                                                        starting_sequence=starting_sequence,
                                                        ncores=n_cores,
                                                        M=M,
                                                        metropolis=metropolis)
        else:

            data, start_seq, W_list = parallel_Od_Langevin(beta,gamma,dt, potential_fn, grad_U_torch, 
                                                        num_points = num_pts,
                                                        starting_sequence=starting_sequence,
                                                        ncores=n_cores,
                                                        M=M,
                                                        metropolis=metropolis)   


    if metropolis:
        # save the acceptance ratio
        p_arr = np.split(Pacc, M)
        Pacc = np.array([np.mean(p) for p in p_arr])
        np.savetxt(os.path.join(folder_path+"Pacc.txt"), Pacc, header="Acceptance ratio", delimiter="\t")



    # Save the data to a file
    
    data = np.array(data).reshape(-1,2)

    #plot the density of the data
    #print("computing kde...")

    #kde = gaussian_kde(data.T)
    #density = kde(data.T)  # probability density 

    density = mc_gaussian_kde_2d(data)

    energies= -np.log(density)/beta # obtain the energy from density distribution

    data_combined = data_combined = [(point[0], point[1], E) for point, E in zip(data, energies)]

    # Save the data to a file
    np.savetxt(os.path.join(folder_path+"data_langevin.txt"), data_combined,
                header="x1 x2 energy", delimiter="\t")


    # plot the density of the data and the potential
    fig, ax = plt.subplots(1,2, figsize=(12, 6))

    plot_hexbin(ax = ax[0], X_data= data[:, 0], Y_data= data[:, 1],
                x1_min=x1_min, x1_max=x1_max,
                x2_min=x2_min, x2_max=x2_max,
                gridsize=gridsize,
                E_data= density, title="Density of the data", colorbar_label="density")
    
    if start_seq is not None:
        for point in start_seq[:]:
            ax[0].plot(point[0], point[1], "ro", markersize=10)

 
    plot_hexbin(ax = ax[1],X_data=data[:, 0],Y_data= data[:, 1],E_data= energies,
                x1_min=x1_min, x1_max=x1_max,
                x2_min=x2_min, x2_max=x2_max,
                gridsize=gridsize,
                title="Potential", colorbar_label="Energy")

    plt.savefig(os.path.join(folder_path+"density_potential.png"), dpi=300)


    # compare the energy result of the Langevin dynamics with the potential

    E_true = potential_fn(torch.tensor(data)).detach().numpy()

    E_diff = E_true - energies

    fig, ax = plt.subplots( figsize=(12, 6))


    if max(E_diff) < 0:
        norm = None
    elif potential_fn == mueller_brown:
        norm = mcolors.TwoSlopeNorm(vmin=min(E_diff), vcenter=0, vmax=max(E_diff))
    else:
        norm = None

    plot_hexbin(X_data=data[:, 0], Y_data= data[:, 1],E_data= E_diff,
                x1_min=x1_min, x1_max=x1_max,
                x2_min=x2_min, x2_max=x2_max,
                gridsize=gridsize,
                title="Energy differences", colorbar_label="", cmap="RdBu", norm = norm)

    plt.savefig(os.path.join(folder_path+"energy_diff.png"), dpi=300)


