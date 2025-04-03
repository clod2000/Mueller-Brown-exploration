####################################################################################################
# 24.07.2025
#
#  library for the toy model
#
#  Colturi Claudio 
# 
####################################################################################################


import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import matplotlib.colors as mcolors
from torch.autograd import grad as Grad  # For gradient computation

import multiprocessing as mp
from scipy.stats import gaussian_kde



# "2D Gaussian" potential function
def gaussian_potential(Q):
    """2D Gaussian potential function for a single point or a grid of points.
    
    Q: Tensor of shape (..., 2), where last dimension contains (x, y) coordinates.
    Returns: Tensor of shape (...) with potential values.
    """
    x, y = Q[..., 0], Q[..., 1]  # Extract x, y from the last dimension
    return - torch.exp(- (x**2 + y**2))  # Gaussian potential





# Mueller-Brown potential 

def mueller_brown(Q):
    """Müller-Brown potential function for a single point or a grid of points.
    
    Q: Tensor of shape (..., 2), where last dimension contains (x, y) coordinates.
    Returns: Tensor of shape (...) with potential values.
    """
    x, y = Q[..., 0], Q[..., 1]  # Extract x, y from the last dimension
    
    # Müller-Brown parameters
    A = torch.tensor([-200, -100, -170, 15], dtype=torch.float32)
    a = torch.tensor([-1, -1, -6.5, 0.7], dtype=torch.float32)
    b = torch.tensor([0, 0, 11, 0.6], dtype=torch.float32)
    c = torch.tensor([-10, -10, -6.5, 0.7], dtype=torch.float32)
    X = torch.tensor([1, 0, -0.5, -1], dtype=torch.float32)
    Y = torch.tensor([0, 0.5, 1.5, 1], dtype=torch.float32)

    # Compute potential using broadcasting: Sum over 4 terms
    U = (A * torch.exp(
        a * (x[..., None] - X)**2 + 
        b * (x[..., None] - X) * (y[..., None] - Y) + 
        c * (y[..., None] - Y)**2
    )).sum(dim=-1)

    return U  # Shape matches input grid


#####################################
#
# plot functions
#

def generate_grid(x1_min, x1_max, x2_min, x2_max, size=100):
    """
    Generate a grid of points in 2D space.

    Parameters:
    - x1_min, x1_max: Range for the first dimension.
    - x2_min, x2_max: Range for the second dimension.
    - size: Number of points in each dimension for the grid.
    """

    x1 = torch.linspace(x1_min, x1_max, size)
    x2 = torch.linspace(x2_min, x2_max, size)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing="ij")
    grid = torch.stack([grid_x1, grid_x2], dim=-1)
    x = grid.reshape((-1, 2))

    return x


def plot_potential(U, x1_min =-2., x1_max= 1., x2_min =-0.5 , x2_max = 2.0, grid_size=100, levels=50, cut = 100, save_path = None):

    """
    Plot the potential energy surface.
    
    Parameters:
    - U: Potential function to be plotted.
    - x1_min, x1_max: Range for the first dimension.
    - x2_min, x2_max: Range for the second dimension.
    - grid_size: Number of points in each dimension for the grid.
    - levels: Number of contour levels.
    - cut: Cutoff value for the potential.
    -save_path: Path to save the plot (if provided).
    """
    # Create a grid of points
    grid_x = generate_grid(x1_min, x1_max, x2_min, x2_max, grid_size)
    
    # Compute potential values on the grid
    U_values = U(grid_x).reshape(grid_size, grid_size).detach().numpy()
    U_cut = U_values.copy()
    U_cut[U_values > cut] = cut
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    

    contour0 = axes[0].contourf(grid_x[:, 0].reshape((grid_size, grid_size)), 
                        grid_x[:, 1].reshape((grid_size, grid_size)), 
                        U_values, 
                        levels=levels,
                        cmap=cm.viridis_r)


    cbar = plt.colorbar(contour0, ax=axes[0])
    cbar.set_label("potential")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    axes[0].set_title("Mueller-Brown potential")


    contour1 = axes[1].contourf(grid_x[:, 0].reshape((grid_size, grid_size)), 
                        grid_x[:, 1].reshape((grid_size, grid_size)), 
                        U_cut, 
                        levels=levels,
                        cmap=cm.viridis_r)

    cbar = plt.colorbar(contour1, ax=axes[1], shrink=0.85)
    cbar.set_label("potential")
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_2$")
    axes[1].set_title(f"cutted potential (U>{cut})")
    axes[1].set_aspect("equal")


    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()



# this method doesn't save the figure

def plot_hexbin(X_data, Y_data, E_data, title = "Hexbin plot",
                x1_min =-2., x1_max= 1., x2_min =-0.5 , x2_max = 2.0, 
                ax=None, colorbar_label = "", gridsize=50, cmap =cm.viridis_r, norm=None):
    """
    Plot hexbin of the data points.
    
    Parameters:
    - X_data: X-coordinates of the data points.
    - Y_data: Y-coordinates of the data points.
    - E_data: Energy values of the data points.
    - x1_min, x1_max: Range for the first dimension.
    - x2_min, x2_max: Range for the second dimension.
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    

    hb = ax.hexbin(X_data, Y_data, C=E_data, gridsize=gridsize, cmap=cmap, mincnt=1, norm=norm)
    cb = plt.colorbar(hb, ax=ax, label=colorbar_label, shrink=0.85)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_aspect("equal")
    ax.set_title(title)


 

#########################################
#
# Functions for Langevin dynamics
#
#########################################   


# Metropolis step to satisfy detailed balance
#
# since we are sampling as x(i) = x(i-1) - D * beta * dt * gradU + Wi, and Wi = sqrt(2Ddt) * N(0,1)
#  basically x(i) is sampled from a gaussian distribution
# centered in  mu(x(i-1)) = x(i-1) - D * beta * dt * gradU and with variance 2Ddt
# so the transition probability T(x(i) | x(i-1)) is proportional to exp[ - (x(i) - mu(x(i-1)))^2 / (4Ddt)]

def metropolis_step(x, x_new, beta, D, dt, grad, grad_new, U):
    """
    Perform a Metropolis step to satisfy detailed balance.
    
    Parameters:
    - x: Current position.
    - x_new: Proposed new position.
    - beta: Inverse temperature.
    - D: Diffusion coefficient.
    - dt: Time step.
    - grad: Gradient of the potential at the current position.
    - grad_new: Gradient of the potential at the proposed new position.
    - U: Potential function.

    Returns:
    - x_new: Accepted new position.
    """

    def mu(x,gradient):
        return x - D * beta * dt * gradient
    
    var =  2 * D * dt

    U_new = U(torch.tensor(x_new)).detach().item()
    U_old = U(torch.tensor(x)).detach().item()

    #print("U_new:", U_new, "U_old:", U_old)
    #print("grad_new:", grad_new, "grad:", grad)

    Pnum = np.exp(- np.linalg.norm(x - mu(x_new,grad_new))**2 / (2* var)) * np.exp(-beta*U_new)
    Pden = np.exp(- np.linalg.norm(x_new - mu(x,grad))**2 / (2* var)) * np.exp(-beta*U_old)

    #print("Pnew:", Pnew," ", np.exp(-beta*U_old)," ", mu(x,grad), "\t", "Pold:", Pold," ", np.exp(-beta*U_new)  )
    

    P_accept = min(1, Pnum / Pden )

    
    if np.random.rand(1) < P_accept:
        return x_new, 1
    else:
        return x,0
  


def Od_Langevin(beta,gamma,dt, U, grad_U, num_points = 10000, starting_sequence = None, DIM = 2, metropolis = True):

    """
    Simulate  Overdamped Langevin dynamics

    Parameters:
    - beta: Inverse temperature.
    - gamma: Friction coefficient.
    - dt: Time step.
    - U: Potential function.
    - grad_U: Gradient function.
    - num_points: Number of points to simulate for each Langevin trajectory.
    - starting_sequence: Initial points for Langevin dynamics:
            if None, M random points are generated in the range [-2, 2].
    - DIM: Dimension of the Langevin dynamics (default is 2 for 2D potential).
    - metropolis: If True, use Metropolis step to satisfy detailed balance.
                  If False, use Langevin dynamics without Metropolis step.

    Returns:
    - x_list: List of Langevin trajectories.
    - starting_sequence: Starting points for Langevin dynamics.
    - W_list: List of random kicks.
    """

    if metropolis:
        print("------------------------")
        print("Using Langevin dynamics with Metropolis step.")
        print("------------------------")


    D =1/ (beta *gamma) #diffusion coefficient
   
    if starting_sequence is None:
        M = 10
        starting_sequence = np.random.uniform(-2,2, (M, DIM))
    else:
        M = starting_sequence.shape[0]

    x_list = []
    W_list = []

    #print("computing Langevin dynamics... [", end="")
    Paccept = []

    for j in range(M):
        #print("Simulating Langevin dynamics for trajectory", j+1, "of", M)
        #print("*", end = "")
        x = np.zeros( (num_points, DIM) )
        x[0] = starting_sequence[j]
    
        for i in range(1, num_points):
            
            #sample the random kick from a normal distribution
            W = np.random.normal( 0,1,DIM) * np.sqrt(2 * D * dt)
            W_list.append(W)

            Q = torch.tensor(x[i-1], dtype=torch.float32)
            grad = grad_U(U, Q)
            # convert grad to numpy array
            grad = grad.detach().numpy()
            #print("Gradient of U at (",Q.detach().numpy()[0],",",Q.detach().numpy()[1],"):", grad)

            x_new = x[i-1] - D * beta * dt * grad + W

            if metropolis:
                grad_new = grad_U(U, torch.tensor(x_new, dtype=torch.float32)).detach().numpy()
                x[i], Pacc = metropolis_step(x[i-1], x_new, beta, D, dt, grad,grad_new, U)
                Paccept.append(Pacc)
            else:
                x[i] = x_new

        x_list.append(x)
    
    if metropolis:
        print("------------------------")
        print("Acceptance rate:", np.mean(Paccept))
        print("------------------------")

        return x_list, starting_sequence, W_list, Paccept
    
    else:
        return x_list, starting_sequence, W_list
          

#################################################
# 
# Functions for computing gradients
# 
# ###############################################         


def grad_U(U, Q, epsilon=1e-6):
    grad_U = np.zeros_like(Q)
    for i in range(len(Q)):
        Q_plus = Q.detach().clone()
        Q_minus = Q.detach().clone()
        Q_plus[i] += epsilon
        Q_minus[i] -= epsilon
        grad_U[i] = (U(torch.tensor(Q_plus)) - U( torch.tensor(Q_minus))) / (2 * epsilon)
    return grad_U


#################

def grad_U_torch(U, Q):
    """
    Computes the gradient of the potential function U(Q) using PyTorch autograd.
    
    Parameters:
    - U: Callable function that takes a PyTorch tensor Q and returns a scalar potential.
    - Q: PyTorch tensor of shape (N,) where N is the number of variables.

    Returns:
    - grad_U: PyTorch tensor of shape (N,) containing the gradient of U at Q.
    """
    Q = Q.clone().detach().requires_grad_(True)  # Ensure Q requires gradients
    potential = U(Q)  # Compute potential energy (must return a scalar)
    
    if potential.dim() != 0:
        raise ValueError("Potential function U(Q) must return a scalar.")

    potential.backward()  # Compute gradients
    return Q.grad  # Return the computed gradient


import torch


#############################
#
# PARALLEL FUNCTIONS
#
###############################



# Faster kde (multicore)

import numpy as np
import multiprocessing as mp
from scipy.stats import gaussian_kde


# Worker function: Creates its own KDE model
def kde_eval(chunk, data, ):
    kde = gaussian_kde(data.T)  # Recreate KDE in each process
    return kde(chunk.T)  # Evaluate density

def mc_gaussian_kde_2d(data, num_cores=None):
    # Ensure data is 2D: shape (N, 2)
    if data.shape[1] != 2:
        raise ValueError("Data must have shape (N, 2) where N is the number of points.")

    # Set number of cores
    if num_cores is None:
        num_cores = mp.cpu_count()

    print(f"Using {num_cores} cores for parallel computation of kde.")
   
    # Split grid points into chunks
    chunks = np.array_split(data, num_cores)  # Now shape (num_chunks, num_points_per_chunk, 2)

    # Run parallel computation using `starmap`
    with mp.Pool(num_cores) as pool:
        results = pool.starmap(kde_eval, [(chunk, data) for chunk in chunks])

    # Combine results
    final_density = np.concatenate(results) # Reshape to match grid

    return final_density  # Return grid and KDE values




def parallel_Od_Langevin(beta,gamma,dt, U, grad_U, num_points = 10000,starting_sequence = None, ncores=None, M=5, DIM = 2, metropolis = True):

    """"
    "Simulate  Overdamped Langevin dynamics"

    Parameters:
    - beta: Inverse temperature.
    - gamma: Friction coefficient.
    - dt: Time step.
    - U: Potential function.
    - grad_U: Gradient function.
    - num_points: Number of points to simulate for each Langevin trajectory.
    - starting_sequence: Initial points for Langevin dynamics:
            if None, M random points are generated in the range [-2, 2].
    - ncores: Number of cores to use for parallel computation: 
            if None, use all available cores.
    - M: Number of Langevin trajectories to simulate: 
            if starting_sequence is not None,  M = starting_sequence.shape[0]/DIM
            (default is 5).
    - DIM: Dimension of the Langevin dynamics (default is 2 for 2D potential).
    - metropolis: If True, use Metropolis step to satisfy detailed balance.

    Returns:
    - x_list: List of Langevin trajectories.
    - starting_sequence: Starting points for Langevin dynamics.
    - W_list: List of random kicks.

    """

    if starting_sequence is None:
        starting_sequence = np.random.uniform(-2,2, (M, DIM))

    else:
        M = starting_sequence.shape[0]
        print("M:", M)

    if ncores is None:
        ncores = int(np.min(mp.cpu_count(),M))  # Use all available cores or M, whichever is smaller

    if ncores > M:
        print("Number of cores exceeds number of Langevin trajectories. Using M cores.")
        ncores =int(M)

    print("Starting Langevin dynamics with", ncores, "cores")


    # example. ncores = 2, M = 5
    #  first core 3 processes, second core 2 processes
    #  core0 proc1 core1 proc2 core0 proc3 cor1 proc4 core0 proc5
  

    base_size = M // ncores  # Base size of each chunk
    extra = M % ncores  # Remaining points to distribute
   
    # Creazione della lista delle dimensioni
    chunks_size = [base_size + 1 if i < extra else base_size for i in range(ncores)]
    
    chunks = np.split(starting_sequence, np.cumsum(chunks_size)[:-1])

     
    #print("chunks: ", chunks)
    print( "chunk sizes:")
    for i in range(ncores):
        print(f"core {i+1}: ",chunks[i].shape[0], "points")


    with mp.Pool(ncores) as pool:
        results = pool.starmap(Od_Langevin, [(beta, gamma, dt, U, grad_U, num_points, chunks[i],2,metropolis ) for i in range(ncores)])
       
    # Combine results
    
    x_list = []
   
    for result in results:
        x_list.append([point for point in result[0]])
      
        

    print("-----------------------")
    
    

    # The following line is taken from chatgpt, honestly I have no idea how it works
    # but it seems to work ( it didn't work)
    #ordered_list = np.vstack([np.vstack(gruppo) for gruppo in x_list])[np.argsort(np.vstack([np.vstack(gruppo) for gruppo in x_list])[:, 0])].tolist()
    
    # Concatenate all trajectories into one large array
    all_points_list = []
    for result in results:
        # result[0] is the list of trajectories from one core
        for trajectory in result[0]:
            all_points_list.append(trajectory)
    # Concatenate all trajectories into one large array
    all_points = np.concatenate(all_points_list, axis=0)
    # W_list might need similar concatenation if needed
    all_W = np.concatenate([res[2] for res in results if res[2] is not None], axis=0) # Example
    if metropolis:
        # Concatenate acceptance probabilities if available
        # Note: This assumes that all results have the same length for acceptance probabilities
        # If not, you might need to handle this differently
        all_Pacc = np.concatenate([res[3] for res in results if res[3] is not None], axis=0)
        return all_points, starting_sequence, all_W, all_Pacc # Return the combined array
    
    else:
        # Return combined data without acceptance probabilities
        return all_points, starting_sequence, all_W



    #return ordered_list, starting_sequence, results[0][2]

