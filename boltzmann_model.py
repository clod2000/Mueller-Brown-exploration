##############################################################################
#
#  Boltzmann Model 
#
#############################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RestrictedBoltzmannMachine(nn.Module):

    def __init__(self, n_visible, n_hidden,device = "cpu", k=1):
        

        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k # number of repetitions in contrastive divergence
        self.device = device
        self.sigma = torch.tensor(1.0).to(device)  # standard deviation for Gaussian sampling for visible units
        self.lr = 0.01  # learning rate for weight updates

        # Initialize weights and biases
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) *np.sqrt(2./(n_visible + n_hidden)))  # Glorot/Xavier initialization
        self.v_bias = nn.Parameter(torch.zeros(n_visible))  
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))  
    
    def  P_h_given_v (self,v):
        # compute P(h = 1|v)
        return torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)

    def mu_v_given_h(self,h):  
        # compute mu(v|h)
        return torch.matmul(h, self.W.t()) + self.v_bias
    
    def sample_h(self, v):
        # Sample hidden layer given visible layer
        p_h = self.P_h_given_v(v)
        return torch.bernoulli(p_h), p_h
    
    def sample_v(self, h):
        # Sample visible layer given hidden layer
        mu = self.mu_v_given_h(h)
        noise = torch.normal(0, 1, size=mu.size(), device=self.device)  # Gaussian noise
        v = mu + noise*self.sigma
        return v, mu


    # Here make sense to divide the training into more functions to better adapt to batch training
    # and to add the possibility of changing the learning rate, adding weight decay, etc.



    def contrastive_divergence(self, v0):
        # Contrastive Divergence algorithm
        
        # Step 1: Positive phase
        # Sample hidden layer h0 from visible layer v0

        h0_samples, h0_probs = self.sample_h(v0) # Sample hidden layer from visible layer



        

        # Step 2: Negative phase ( Gibbs sampling)
        # compute mean of visible layer given hidden layer

        #start the chain with samples from the positive phase
        v_chain = v0  # used only if k = 0
        h_chain = h0_samples 

        for _ in range(self.k):
            v_chain, _ = self.sample_v(h_chain)  # Sample visible layer from hidden layer
            h_chain, h_chain_probs = self.sample_h(v_chain)  # Sample hidden layer from visible layer

        return v0, h0_probs, v_chain.detach(), h_chain_probs.detach()  # Detach to avoid backpropagation through the chain
        

    def compute_gradients(self, v0, h0_probs, vk, hk_probs):

        # v0 is the batch used
        pos_statistics = torch.matmul(v0.t(), h0_probs)  # Positive statistics, using probabilities instead of
                                                         # samples is usually better

        neg_statistics = torch.matmul(vk.t(), hk_probs)  # Negative statistics


        grad_W = (pos_statistics - neg_statistics) / v0.size(0)  # Gradient for weights
        grad_v_bias = torch.mean(v0 - vk, dim=0)  # Gradient for visible bias
        grad_h_bias = torch.mean(h0_probs - hk_probs, dim=0) # Gradient for hidden bias using probabilities
      

        return grad_W, grad_v_bias, grad_h_bias
    

    def reconstruct(self, v_in, n_gibbs=1):
        """
        Reconstructs visible data using the RBM. Typically returns the mean
        of the reconstructed distribution P(v|h) after n_gibbs steps.

        Args:
            v_in (Tensor): Input visible data (batch_size, n_visible).
            n_gibbs (int): Number of Gibbs sampling steps v->h->v.

        Returns:
            Tensor: Mean of the reconstructed visible distribution.
        """
        v_rec = v_in
        h_state = torch.zeros(v_in.shape[0], self.n_hidden, device=v_in.device)

        # Gibbs sampling loop
        for _ in range(n_gibbs):
            h_state, _ = self.sample_h(v_rec)
            v_rec, _ = self.sample_v(h_state) # Sample v for next step

        # Final reconstruction: Calculate mean based on last hidden state
        # It's common to use the sampled h_state for the final pass
        _, v_mean_rec = self.sample_v(h_state)
        # Or could use final hidden probs:
        # _, h_probs_final = self._sample_h_given_v(v_rec)
        # _, v_mean_rec = self._sample_v_given_h(h_probs_final)

        return v_mean_rec
    


 