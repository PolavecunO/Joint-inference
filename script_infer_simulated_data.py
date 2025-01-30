# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:10:35 2025

@author: Pol
"""

import matplotlib.pyplot as plt
import numpy as np
from inference.infer_abundance_MLE import infer_abundance_MLE
from inference.infer_abundance_MLE_complex_theta import infer_abundance_MLE_complex_theta
from inference.infer_abundance_MCMC import infer_abundance_MCMC
from inference.infer_abundance_MLE_theta_tau import infer_abundance_MLE_theta_tau
from data_simulation_treatment.simulate_abundance_data import simulate_abundance_data

# === Paramètres de simulation ===
N0_real = 10
N1_real = 1000
T = 193
tau_real = 10
theta_values = [1.5, 0.01]
theta_change_times = [42]
theta_transition_type = "sigmoid"
missing_data = False
p = 0.4
Gamma = False
beta_gamma = 0.1
method = "all"

num_samples = 1000
epsilon = 1e-6
seuil_correction = 0.034
alpha_prior = 6.25
beta_prior = 0.01

y = simulate_abundance_data(N0_real, N1_real, T, tau_real, theta_values, theta_change_times, missing_data, p, Gamma, beta_gamma, random_seed=42)

results = {}

try:
    if method in ["MLE", "all"]:
        results["MLE"] = infer_abundance_MLE(y, T, epsilon, seuil_correction, alpha_prior, beta_prior, theta_values, theta_change_times)
    if method in ["MLE_complex_theta", "all"]:
        results["MLE_complex_theta"] = infer_abundance_MLE_complex_theta(y, T, epsilon, seuil_correction, alpha_prior, beta_prior, theta_values, theta_change_times, theta_transition_type)
    if method in ["MLE_theta_tau", "all"]:
        results["MLE_theta_tau"] = infer_abundance_MLE_theta_tau(y, T, epsilon, seuil_correction, alpha_prior, beta_prior, max_theta_segments=len(theta_values))
    if method in ["MCMC", "all"]:
        trace, _ = infer_abundance_MCMC(y, T, num_samples, epsilon, alpha_prior, beta_prior, alpha_prior, beta_prior)
        tau_post = trace.posterior["tau"].values.flatten()
        results["MCMC"] = int(np.median(tau_post))
except Exception as e:
    print(f"Erreur lors de l'inférence ({method}) : {e}")
    exit()

# === Affichage des résultats ===
plt.figure(figsize=(12, 6))
plt.plot(range(T), y, label="Observations", marker="o", linestyle="None", color="black")

for name, res in results.items():
    tau_est = res[0] if isinstance(res, tuple) else res
    estimated_sizes = [res[2] if t < tau_est else res[4] for t in range(T)]
    plt.plot(range(T), estimated_sizes, linestyle="--", label=f"Taille inférée ({name})")

plt.xlabel("Temps")
plt.ylabel("Nombre d'observations / Taille estimée")
plt.title("Estimation des tailles sur données simulées")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
