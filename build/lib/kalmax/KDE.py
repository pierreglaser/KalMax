from typing import Callable

import jax.numpy as jnp
from jax import vmap, jit
from kalmax.utils import gaussian_pdf
from kalmax.kernels import gaussian_kernel

from functools import partial

def kde(
        bins: jnp.ndarray,
        trajectory: jnp.ndarray,
        spikes: jnp.ndarray,
        kernel,
        kernel_kwargs: dict = {},
        ) -> jnp.ndarray:
    """
    Performs KDE to estimate the expected number of spikes each neuron will fire at each position in `bins` given past `trajectory` and `spikes` data. This estimate is an expected-spike-count-per-timebin, in order to get firing rate in Hz, divide this by dt.

    Kernel Density Estimation goes as follows (the denominator corrects for for non-uniform position density): 

              # spikes observed at x     sum_{spike_times} K(x, x(ts))     Ks
      mu(x) = ---------------------- ==> ----------------------------- :=  --
                  # visits to x            sum_{all_times} K(x, x(t))      Kx
              = exp[log(Ks) - log(Kx)]
    
    Parameters
    ----------
    bins : jnp.ndarray, shape (N_bins, D,)
        The position bins at which to estimate the firing rate
    trajectory : jnp.ndarray, shape (T, D)
        The position of the agent at each time step
    spikes : jnp.ndarray, shape (T, N_neurons)
        The spike counts of the neuron at each time step (integer array, can be > 1)
    kernel : function
        The kernel function to use for density estimation. See `kernels.py` for signature and examples.
    kernel_kwargs : dict, default={}
        Any additional arguments to the kernel function
    
    Returns
    -------
    kernel_density_estimate : jnp.ndarray, shape (N_neurons, N_bins)
    """
    assert bins.ndim == 2
    assert trajectory.ndim == 2
    assert spikes.ndim == 2

    D = bins.shape[1] # dimension of the space
    
    # vmap the kernel K(x,mu,sigma) so it takes in a vector of positions and a vector of means
    kernel = partial(kernel, **kernel_kwargs)
    vmapped_kernel = vmap(kernel, in_axes=(0, None))
    vmapped_kernel = vmap(vmapped_kernel, in_axes=(None, 0))

    # Pairwise kernel values for each trajectory-bin position pair. The bulk of the computation is done here. 
    kernel_values = vmapped_kernel(trajectory, bins) # size (N_bins, T,)
    
    # Calculate normalisation position density 
    position_density = kernel_values.sum(axis=1)+1e-3 # unnormalised size (N_bins,)
    
    # calculate spike density 
    spike_density = kernel_values @ spikes # size (N_x, N_neurons,)
    spike_density = jnp.where(jnp.isnan(spike_density), 0, spike_density) # replace nans from no-spikes with 0
    
    # calculate kde at each bin position 
    kernel_density_estimate = jnp.exp(jnp.log(spike_density) - jnp.log(position_density)[:,None]).T

    return kernel_density_estimate


def poisson_log_likelihood(spikes, mean_rate):
        """Takes an array of spike counts and an array of mean rates and returns the log-likelihood of the spikes given the mean rate of the neuron (it's receptive field).

        P(X|mu) = (mu^X * e^-mu) / X!
        log(P(X|mu)) = sum_{neurons} [ X * log(mu) - mu - log(X!) ]
        where 
        log(X!) = log(sqrt(2*pi)) + (X+0.5) * log(X) - X    (manually correcting for when X=0) #in fact I don't think this stirling approximation is necessary for speed up
        
        Parameters
        ----------
        spikes : jnp.ndarray, shape (T, N_neurons,)
            How many spikes the neuron actually fired at each bin (int, can be > 1)
        mean_rate : jnp.ndarray, shape (N_neurons, N_bins,)
            The mean rate of the neuron (it's place field) at each bin. This is how many spikes you would _expect_ in at this position in a time dt.
            
        Returns
        -------
        log_likelihood : jnp.ndarray, shape (T, N_bins,)
            The log-likelihood (summed over neurons) of the spikes given the mean rate of the neuron
        """
        spikes_ = jnp.where(spikes, spikes, 1) #replace 0s with 1s as 0! == 1! 
        log_spikecount_factorial = jnp.log(jnp.sqrt(2*jnp.pi)) + (spikes_+0.5)*jnp.log(spikes_) - spikes_
        # log_spikecount_factorial = jnp.log(jnp.math.factorial(spikes)) # TODO test if this is as fast
        logPXmu = spikes @ jnp.log(mean_rate+1e-10) - jnp.sum(mean_rate,axis=0)[None,:] - jnp.sum(log_spikecount_factorial, axis=1)[:,None]

        return logPXmu