import jax.numpy as jnp
from jax import random

import sys
sys.path.insert(0, '../../')

from graph_new import ErdosReniDAGDistribution, ScaleFreeDAGDistribution, UniformDAGDistributionRejection
from dibs.graph_utils import graph_to_mat

from linearGaussian_new import *

from typing import Any, NamedTuple


class Data(NamedTuple):
    """ NamedTuple for structuring simulated synthetic data and their ground
    truth generative model

    Args:
        passed_key (ndarray): ``jax.random`` key passed *into* the function generating this object
        n_vars (int): number of variables in model
        n_observations (int): number of observations in ``x`` and used to perform inference
        n_ho_observations (int): number of held-out observations in ``x_ho``
            and elements of ``x_interv`` used for evaluation
        g (ndarray): ground truth DAG
        theta (Any): ground truth parameters -> it is going to be a list having all the different parameter values.
        x (ndarray): i.i.d observations from the model of shape ``[n_observations, n_vars]``

    """

    passed_key: Any

    n_vars: int
    n_observations: int

    g: Any
    theta: Any
    x: Any


def make_synthetic_bayes_net(*,
    key,
    n_vars,
    graph_dist,
    generative_model,
    n_observations=100
):
    """
    Returns an instance of :class:`~dibs.metrics.Target` for evaluation of a method on
    a ground truth synthetic causal Bayesian network

    Args:
        key (ndarray): rng key
        n_vars (int): number of variables
        graph_dist (Any): graph model object. For example: :class:`~dibs.models.ErdosReniDAGDistribution`
        generative_model (Any): BN model object for generating the observations. For example: :class:`~dibs.models.LinearGaussian`
        n_observations (int): number of observations generated for posterior inference
        n_ho_observations (int): number of held-out observations generated for evaluation
        n_intervention_sets (int): number of different interventions considered overall
            for generating interventional data
        perc_intervened (float): percentage of nodes intervened upon (clipped to 0) in
            an intervention.

    Returns:
        :class:`~dibs.target.Data`:
        synthetic ground truth generative DAG and parameters as well observations sampled from the model
    """

    # remember random key
    passed_key = key.copy()

    # generate ground truth observations
    
    ## step1: generate the graph, this should be the same in both procedures
    key, subk = random.split(key)
    g_gt = graph_dist.sample_G(subk)
    g_gt_mat = jnp.array(graph_to_mat(g_gt))


    ## step2 -3 : generate parameters, instead of sampling once and keeping the same parameter sample for all the observations we want to create 
    ## we will sample one parameter sample for each sample that we want to create.
    ## on the final step then we will 
    ## see this link for another way to use the random keys : https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html

    observations = jnp.array(object= jnp.empty(shape=(0,n_vars)))
    thetas = []
    for i in range(n_observations):
        key, subk = random.split(key)
        theta = generative_model.sample_parameters(key=subk, n_vars = n_vars)
        thetas.append(theta)
        key, subk = random.split(key)
        observation = generative_model.sample_obs(key=subk, n_samples = 1, g = g_gt, theta = theta)
        observations = jnp.append(observations, values= observation, axis=0)

    # return and save generated target object
    data = Data(
        passed_key=passed_key,
        n_vars=n_vars,
        n_observations=n_observations,
        g=g_gt_mat,
        theta=thetas,
        x=observations)
    
    return data

def make_graph_model(*, n_vars, graph_prior_str, edges_per_node=2):
    """
    Instantiates graph model

    Args:
        n_vars (int): number of variables in graph
        graph_prior_str (str): specifier for random graph model; choices: ``er``, ``sf``
        edges_per_node (int): number of edges per node (in expectation when applicable)

    Returns:
        Object representing graph model. For example :class:`~dibs.models.ErdosReniDAGDistribution` or :class:`~dibs.models.ScaleFreeDAGDistribution`
    """
    if graph_prior_str == 'er':
        graph_dist = ErdosReniDAGDistribution(
            n_vars=n_vars, 
            n_edges_per_node=edges_per_node)

    elif graph_prior_str == 'sf':
        graph_dist = ScaleFreeDAGDistribution(
            n_vars=n_vars,
            n_edges_per_node=edges_per_node)

    else:
        assert n_vars <= 5, "Naive uniform DAG sampling only possible up to 5 nodes"
        graph_dist = UniformDAGDistributionRejection(
            n_vars=n_vars)

    return graph_dist


def make_linear_gaussian_model(*, key, n_vars=20, graph_prior_str='sf', 
    obs_noise=0.1, mean_edge=0.0, sig_edge=1.0, min_edge=0.5, n_observations=100):
    """
    Samples a synthetic linear Gaussian BN instance 

    Args:
        key (ndarray): rng key
        n_vars (int): number of variables
        n_observations (int): number of iid observations of variables
        n_ho_observations (int): number of iid held-out observations of variables
        graph_prior_str (str): graph prior (`er` or `sf`)
        obs_noise (float): observation noise
        mean_edge (float): edge weight mean
        sig_edge (float): edge weight stddev
        min_edge (float): min edge weight enforced by constant shift of sampled parameter

    Returns:
        tuple(:class:`~dibs.models.LinearGaussian`, :class:`~dibs.target.Data`):
        linear Gaussian inference model and observations from a linear Gaussian generative process
    """

    # init models
    graph_dist = make_graph_model(n_vars=n_vars, graph_prior_str=graph_prior_str)

    generative_model = LinearGaussian(
        graph_dist=graph_dist, obs_noise=obs_noise,
        mean_edge=mean_edge, sig_edge=sig_edge,
        min_edge=min_edge)

    inference_model = LinearGaussian(
        graph_dist=graph_dist, obs_noise=obs_noise,
        mean_edge=mean_edge, sig_edge=sig_edge,
        min_edge=min_edge)

    # sample synthetic BN and observations
    key, subk = random.split(key)
    data = make_synthetic_bayes_net(
        key=subk, n_vars=n_vars,
        graph_dist=graph_dist,
        generative_model=generative_model,
        n_observations=n_observations)

    return data, inference_model

