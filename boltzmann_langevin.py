from boltzmann_model import RestrictedBoltzmannMachine
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import sklearn

import matplotlib.pyplot as plt


#-------- Starting parameters ----------

path = "/home/clod/Desktop/Thesis/toy-model/"

"load data according to a certain sumulation"
beta = 0.03
gamma = 0.5
dt = 5e-4
num_pts = 700000
metropolis = True
mueller_brown_p = True



def standardize_data(X):
    """Standardizes data to have mean 0 and std dev 1."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0 # Avoid division by zero
    X_std = (X - mean) / std
    return X_std, mean, std


def train_RBM(rbm,dataset,epochs = 10, batch_size = 32, lr = 1e-3, weight_decay = 1e-5):
    """Train the RBM using the given dataset."""

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(rbm.parameters(), lr=lr, weight_decay=weight_decay) # Adam optimizer with weight decay

    rbm.to(rbm.device)
    rbm.train() 

    history = {'epoch': [], 'reconstruction_error': []}
    print(f"Training RBM for {epochs} epochs with batch size {batch_size} and learning rate {lr}")

    for epoch in range(epochs):
        epoch_error = 0.0
        n_batches = 0

        for batch_idx,batch in enumerate(dataloader):
            batch = batch[0].to(rbm.device)
           
            # perform contrastive-divergence
            v0, h0_probs, vk, hk_probs = rbm.contrastive_divergence(batch)
        
            # Compute the gradients
            grad_W, grad_v_bias, grad_h_bias = rbm.compute_gradients(v0, h0_probs, vk, hk_probs)

            # Update the weights and biases manually
            # (using minus because we want to maximize the likelihood)

                  
# INSIDE train_RBM function, AFTER computing gradients and BEFORE zero_grad/assignment:

            # --- DEBUGGING: Print gradient norms ---
            if batch_idx % 50 == 0: # Print every 50 batches
                norm_W = torch.linalg.norm(grad_W).item()
                norm_vb = torch.linalg.norm(grad_v_bias).item()
                norm_hb = torch.linalg.norm(grad_h_bias).item()
                # Also print parameter norms maybe
                norm_W_param = torch.linalg.norm(rbm.W.data).item()
                print(f"  Epoch {epoch+1}, Batch {batch_idx}: |grad_W|={norm_W:.2e}, |grad_vb|={norm_vb:.2e}, |grad_hb|={norm_hb:.2e}, |W|={norm_W_param:.2f}")
            # --- END DEBUGGING ---

            # Zero gradients before assignment (good practice with optimizers)
            optimizer.zero_grad()

    
            if rbm.W.requires_grad:
                rbm.W.grad = -grad_W # Negative sign because optimizer minimizes
            if rbm.v_bias.requires_grad:
                rbm.v_bias.grad = -grad_v_bias
            if rbm.h_bias.requires_grad:
                rbm.h_bias.grad = -grad_h_bias



            #optimizer step()
            optimizer.step()

            # compute the reconstruction error
            with torch.no_grad():
                v0_reconstructed, _ = rbm.sample_v(h0_probs)
                batch_error = torch.mean((batch - v0_reconstructed) ** 2).item()
                epoch_error += batch_error
            n_batches += 1

        avg_epoch_error = epoch_error / n_batches
        history['epoch'].append(epoch + 1)
        history['reconstruction_error'].append(avg_epoch_error)
        print(f"Epoch {epoch+1}/{epochs}, Avg Reconstruction Error (MSE): {avg_epoch_error:.6f}")

    print("Training finished.")
    rbm.eval() # Set model to evaluation mode
    return history

            






if __name__ == "__main__":


    if metropolis:
        path += "langevin_metropolis/"

        if mueller_brown_p:
            path += "mueller_brown/"
        else:
            path += "gaussian_potential/"

    else:
        path += "langevin/"


    # path example:/home/clod/Desktop/Thesis/toy-model/langevin/beta=0.05_gamma=0.5_dt=1e-05
    # /home/clod/Desktop/Thesis/toy-model/langevin/beta=0.05_gamma=0.5_dt=0.0001



    directory = path + "beta=" + str(beta) + "_gamma=" + str(gamma) + "_dt=" + str(dt) + "/nmpts=" + str(num_pts) + "/"
    print("Loading data from: ", directory)
    if not os.path.exists(directory):
        sys.exit("Directory does not exist")

    X,Y,energy = np.loadtxt(os.path.join(directory,"data_langevin.txt"), skiprows=1, unpack=True)
    raw_data = np.column_stack((X,Y))


        # --- Data Preparation ---
    def create_gaussian_blobs_data(n_samples=1000, n_features=10, centers=3, std=0.5):
        """Generates sample data resembling Gaussian blobs."""
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,
                        cluster_std=std, random_state=42)
        return X.astype(np.float32)

        # 1. Generate and Prepare Data
    N_SAMPLES = 200000
    N_FEATURES = 2 # Dimensionality of your simulation data
    raw_data = create_gaussian_blobs_data(n_samples=N_SAMPLES, n_features=N_FEATURES, centers=2, std=0.5)





    

    # !!! Crucial: Standardize Data !!!
    # Continuous RBM assumes visible units follow N(mean, sigma^2) where sigma is often 1.
    # Standardizing to mean=0, std=1 makes this assumption valid.

    standardized_data, data_mean, data_std = standardize_data(raw_data)
    print("Standardized data shape:", standardized_data.shape)
    #print("Sample standardized data point:", standardized_data[0, :5])
    print(f"Mean after standardization (approx 0): {np.mean(standardized_data, axis=0)[:5]}")
    print(f"Std after standardization (approx 1): {np.std(standardized_data, axis=0)[:5]}")

    # Convert to PyTorch tensors
    tensor_data = torch.tensor(standardized_data, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_data)


    # 2. Create and Train the RBM
    N_VISIBLE = 2
    N_HIDDEN = 1000 # Number of hidden features to learn
    K_CD =  1  # Contrastive Divergence steps
    LEARNING_RATE = 1E-7
    # EPOCHS = 10   
    EPOCHS = 60
    BATCH_SIZE = 128 
    WEIGHT_DECAY = 1 #e-2


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    rbm_model = RestrictedBoltzmannMachine(n_visible=N_VISIBLE, n_hidden=N_HIDDEN,
                                            device=DEVICE, k=K_CD)


    rbm_model.to(DEVICE)

    training_history = train_RBM(rbm_model, dataset, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Save the model
    torch.save(rbm_model.state_dict(), os.path.join(directory, "rbm_model.pth"))

     # Plot training error
    plt.figure()
    plt.plot(training_history['epoch'], training_history['reconstruction_error'])
    plt.xlabel("Epoch")
    plt.ylabel("Avg Reconstruction Error (MSE)")
    plt.title("RBM Training Curve")
    plt.grid(True)
    plt.show()

    # 3. Reconstruct some data points
    print("\nReconstructing a few data points...")
    n_reconstruct = 100        # num_pts // 100  # Number of samples to reconstruct
    original_samples_std = standardized_data[:n_reconstruct]
    original_samples_raw = raw_data[:n_reconstruct]

    # Convert samples to tensor and move to device for reconstruction
    samples_tensor = torch.tensor(original_samples_std, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        reconstructed_mean_std = rbm_model.reconstruct(samples_tensor, n_gibbs=1)

    # Move back to CPU and numpy for comparison
    reconstructed_mean_std_np = reconstructed_mean_std.cpu().numpy()

    # Convert reconstructed data back to original scale
    reconstructed_mean_raw = reconstructed_mean_std_np * data_std[:N_VISIBLE] + data_mean[:N_VISIBLE] # Make sure std/mean match features

    print("\nOriginal Raw Samples (first 5 features):")
    print(np.round(original_samples_raw[:, :5], 2))
    print("\nReconstructed Raw Samples (Mean, first 5 features):")
    print(np.round(reconstructed_mean_raw[:, :5], 2))

    # Calculate overall MSE on the sample reconstructions (original scale)
    mse_original_scale = np.mean((original_samples_raw - reconstructed_mean_raw)**2)
    print(f"\nMean Squared Error on {n_reconstruct} samples (original scale): {mse_original_scale:.4f}")

    # Plot original vs reconstructed with the same dimension for the limits
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(original_samples_raw[:, 0], original_samples_raw[:, 1], alpha=0.5, label='Original')

    xmin = min(original_samples_raw[:, 0].min(), reconstructed_mean_raw[:, 0].min()) -1
    xmax = max(original_samples_raw[:, 0].max(), reconstructed_mean_raw[:, 0].max()) +1
    ymin = min(original_samples_raw[:, 1].min(), reconstructed_mean_raw[:, 1].min()) -1
    ymax = max(original_samples_raw[:, 1].max(), reconstructed_mean_raw[:, 1].max()) +1
    plt.xlim(xmin, xmax)    
    plt.ylim(ymin, ymax)
    #plt.scatter(original_samples_std[:, 0], original_samples_std[:, 1], alpha=0.5, label='Original (Standardized)')
    plt.title("Original Samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.scatter(reconstructed_mean_raw[:, 0], reconstructed_mean_raw[:, 1], alpha=0.5, color='orange', label='Reconstructed')
    plt.title("Reconstructed Samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(xmin, xmax)    
    plt.ylim(ymin, ymax)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
  