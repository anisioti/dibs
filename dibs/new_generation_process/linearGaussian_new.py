import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.stats import norm as jax_normal

try:
    from jax.numpy import index_exp as index
except ImportError:
    # for jax <= 0.3.2
    from jax.ops import index
    
class LinearGaussian:
    """
    Linear Gaussian BN model corresponding to linear structural equation model (SEM) with additive Gaussian noise.

    Each variable distributed as Gaussian with mean being the linear combination of its parents
    weighted by a Gaussian parameter vector (i.e., with Gaussian-valued edges).
    The noise variance at each node is equal by default, which implies the causal structure is identifiable.

    Args:
        graph_dist: Graph model defining prior :math:`\\log p(G)`. Object *has to implement the method*:
            ``unnormalized_log_prob_soft``.
            For example: :class:`~dibs.graph.ErdosReniDAGDistribution`
            or :class:`~dibs.graph.ScaleFreeDAGDistribution`
        obs_noise (float, optional): variance of additive observation noise at nodes
        mean_edge (float, optional): mean of Gaussian edge weight
        sig_edge (float, optional): std dev of Gaussian edge weight
        min_edge (float, optional): minimum linear effect of parent on child

    """

    def __init__(self, *, graph_dist, obs_noise=0.1, mean_edge=0.0, sig_edge=1.0, min_edge=0.5):
        super(LinearGaussian, self).__init__()

        self.graph_dist = graph_dist
        self.n_vars = graph_dist.n_vars
        self.obs_noise = obs_noise
        self.mean_edge = mean_edge
        self.sig_edge = sig_edge
        self.min_edge = min_edge

        self.no_interv_targets = jnp.zeros(self.n_vars).astype(bool)


    def get_theta_shape(self, *, n_vars):
        """Returns tree shape of the parameters of the linear model

        Args:
            n_vars (int): number of variables in model

        Returns:
            PyTree of parameter shape
        """
        return jnp.array((n_vars, n_vars))


    def sample_parameters(self, *, key, n_vars,n_particles=0, batch_size=0):
        """Samples batch of random parameters given dimensions of graph from :math:`p(\\Theta | G)`

        Args:
            key (ndarray): rng
            n_vars (int): number of variables in BN
            n_particles (int): number of parameter particles sampled
            batch_size (int): number of batches of particles being sampled

        Returns:
            Parameters ``theta`` of shape ``[batch_size, n_particles, n_vars, n_vars]``, dropping dimensions equal to 0
        """
        shape = (batch_size, n_particles, *self.get_theta_shape(n_vars=n_vars))
        theta = self.mean_edge + self.sig_edge * random.normal(key, shape=tuple(d for d in shape if d != 0)) #keeps only the dimensions that are not zero!
        #returns an array
        theta += jnp.sign(theta) * self.min_edge
        return theta


    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv=None, mean_distr = 0):
        """Samples ``n_samples`` observations given graph ``g`` and parameters ``theta``

        Args:
            key (ndarray): rng -> this is the initial key, we will use it as a seed for creating the first random split
            n_samples (int): number of samples
            g (igraph.Graph): graph
            theta (Any): parameters
            interv (dict): intervention specification of the form ``{intervened node : clamp value}``

        Returns:
            observation matrix of shape ``[n_samples, n_vars]``
        """
        if interv is None:
            interv = {}
        if toporder is None:
            toporder = g.topological_sorting() #list: #give a topological ordering on the DAG (always exists at least one since we have a DAG)
            #a topological sort is a graph traversal in which each node v is visited only after all its dependencies are visited.

        x = jnp.zeros((n_samples, len(g.vs))) #array with shape (n_samples,20)

        key, subk = random.split(key)
        z = mean_distr + jnp.sqrt(self.obs_noise) * random.normal(subk, shape=(n_samples, len(g.vs))) #obs_noise is the noise variance, random.normal gives N(0,1)

        # ancestral sampling
        for j in toporder: #have nodels according to their topological ordering

            # intervention
            if j in interv.keys():
                x = x.at[index[:, j]].set(interv[j])
                continue

            # regular ancestral sampling
            parent_edges = g.incident(j, mode='in') # id of incoming edge to the specified node
            parents = list(g.es[e].source for e in parent_edges) #source node for the incoming edges (actually the parent)

            if parents: #it can be the case that there are no parents, then the list is going to be empty, move forward
                mean = x[:, jnp.array(parents)] @ theta[jnp.array(parents), j] # @ is matrix multiplication 
                x = x.at[index[:, j]].set(mean + z[:, j]) #not sure about this mean in the sum , it seems to be always zero. 

            else:
                x = x.at[index[:, j]].set(z[:, j])

        return x
