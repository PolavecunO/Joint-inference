# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:12:12 2025

@author: Pol
"""

import pymc as pm
import numpy as np

def infer_abundance_MCMC(y, T, num_samples=1000, epsilon=1e-6, alpha_prior_N=6.25, beta_prior_N=0.01,
                         alpha_prior_theta=2.0, beta_prior_theta=1.0):
    """
    Implémente un modèle bayésien pour inférer les tailles de population (N0, N1),
    les paramètres d'échantillonnage (\(\theta_0, \theta_1\)) et leurs temps de changement (\(\tau, \tau_\theta\)).

    Arguments :
    - y : Données observées (vecteur des observations \([y_1, ..., y_T]\)).
    - T : Nombre total de temps (longueur de y).
    - num_samples : Nombre d'échantillons MCMC (par défaut : 1000).
    - epsilon : Petite valeur pour éviter les divisions par zéro.
    - alpha_prior_N, beta_prior_N : Paramètres du prior Gamma pour \(N_0, N_1\).
    - alpha_prior_theta, beta_prior_theta : Paramètres du prior Gamma pour \(\theta_0, \theta_1\).

    Retourne :
    - trace : Résultat MCMC contenant les postérieurs des paramètres (\(\tau, \tau_\theta, N_0, N_1, \theta_0, \theta_1\)).
    - model : Le modèle PyMC utilisé.
    """
    # Vérification des données
    if len(y) != T:
        raise ValueError("La longueur des données observées (y) doit correspondre à T.")

    with pm.Model() as model:
        # Priors sur les temps de changement
        tau = pm.DiscreteUniform("tau", lower=1, upper=T - 1)  # Temps de changement pour N
        tau_theta = pm.DiscreteUniform("tau_theta", lower=1, upper=T - 1)  # Temps de changement pour theta
    
        # Priors sur les tailles de population
        N0 = pm.Gamma("N0", alpha=alpha_prior_N, beta=beta_prior_N)
        N1 = pm.Gamma("N1", alpha=alpha_prior_N, beta=beta_prior_N)
    
        # Priors sur les paramètres d'échantillonnage
        theta_0 = pm.Gamma("theta_0", alpha=alpha_prior_theta, beta=beta_prior_theta)
        theta_1 = pm.Gamma("theta_1", alpha=alpha_prior_theta, beta=beta_prior_theta)
    
        # Indices de temps
        indices = np.arange(T)
    
        # Modèle conditionnel pour N_t
        N_t = pm.math.switch(indices < tau, N0, N1)
    
        # Modèle conditionnel pour theta_t
        theta_t = pm.math.switch(indices < tau_theta, theta_0, theta_1)
    
        # Vraisemblance
        y_obs = pm.Poisson("y_obs", mu=pm.math.maximum(theta_t * N_t, epsilon), observed=y)
    
        # Exécution de l'inférence MCMC
        trace = pm.sample(num_samples, cores=1, return_inferencedata=True, progressbar=True)



    return trace, model
