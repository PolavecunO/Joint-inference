# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:12:40 2025

@author: Pol
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from inference.infer_abundance_MLE import infer_abundance_MLE
from inference.infer_abundance_MLE_complex_theta import infer_abundance_MLE_complex_theta
from inference.infer_abundance_MCMC import infer_abundance_MCMC
from inference.infer_abundance_MLE_theta_tau import infer_abundance_MLE_theta_tau
from data_simulation_treatment.process_yearly_data import process_yearly_data, invert_time_data

# === Paramètres ===
file_path = "data/Bombus/Bombus_(Bombus)_terrestris_(100598_specimens).csv"
method = "MLE"  # Choisir parmi : "MLE", "MCMC", "MLE_theta_tau", "MLE_complex_theta", "all"
theta_transition_type = "logarithmic"  # Options possibles :
# "constant", "linear", "exponential", "logarithmic", "sigmoid"

num_samples = 1000
epsilon = 1e-6
seuil_correction = 0.034
alpha_prior_N = 6.25
beta_prior_N = 0.1
alpha_prior_theta = 2.0
beta_prior_theta = 1.0
theta_values = [1.5, 0.01]  
theta_change_years = [1971]  

if method not in ["MLE", "MCMC", "MLE_theta_tau", "MLE_complex_theta", "all"]:
    print("Erreur : méthode inconnue.")
    sys.exit()

# === Extraction des données ===
try:
    years, observations = process_yearly_data(file_path)
    count_data = invert_time_data(observations)
except Exception as e:
    print(f"Erreur lors du traitement du fichier {file_path}: {e}")
    sys.exit()

T = len(count_data)

if T == 0:
    print(f"Aucune donnée valide trouvée dans le fichier {file_path}.")
    sys.exit()

# Convertir les années réelles de changement de theta en indices
theta_change_times = [
    T - (len(years) - 1 - np.where(years[::-1] == year)[0][0])
    for year in theta_change_years if year in years
]

y = count_data
results = {}

try:
    if method in ["MLE", "all"]:
        results["MLE"] = infer_abundance_MLE(y, T, epsilon, seuil_correction, alpha_prior_N, beta_prior_N, theta_values, theta_change_times)

    if method in ["MLE_complex_theta", "all"]:
        results["MLE_complex_theta"] = infer_abundance_MLE_complex_theta(
            y, T, epsilon, seuil_correction, alpha_prior_N, beta_prior_N, theta_values, theta_change_times, theta_transition_type
        )

    if method in ["MLE_theta_tau", "all"]:
        results["MLE_theta_tau"] = infer_abundance_MLE_theta_tau(y, T, epsilon, seuil_correction, alpha_prior_N, beta_prior_N, max_theta_segments=len(theta_values))

    if method in ["MCMC", "all"]:
        trace, _ = infer_abundance_MCMC(y, T, num_samples, epsilon, alpha_prior_N, beta_prior_N, alpha_prior_theta, beta_prior_theta)
        tau_post = trace.posterior["tau"].values.flatten()
        results["MCMC"] = int(np.median(tau_post))
except Exception as e:
    print(f"Erreur lors de l'inférence ({method}) : {e}")
    sys.exit()

# === Affichage des résultats ===
print("\n=== Résultats des inférences ===")
for name, res in results.items():
    print(f"\n--- {name} ---")
    if isinstance(res, tuple):  # Méthodes MLE
        tau_est, alpha_post_N0, beta_post_N0, alpha_post_N1, beta_post_N1, nan_percentage = res
        N0_est = max((alpha_post_N0 - 1) / beta_post_N0, 0) if alpha_post_N0 > 1 else 0
        N1_est = max((alpha_post_N1 - 1) / beta_post_N1, 0) if alpha_post_N1 > 1 else 0
        print(f"  Temps de changement estimé (tau) : {tau_est}")
        print(f"  Taille inférée avant le changement (N0) : {N0_est:.2f}")
        print(f"  Taille inférée après le changement (N1) : {N1_est:.2f}")
        print(f"  Paramètres postérieurs :")
        print(f"    N0 -> alpha = {alpha_post_N0:.2f}, beta = {beta_post_N0:.2f}")
        print(f"    N1 -> alpha = {alpha_post_N1:.2f}, beta = {beta_post_N1:.2f}")
        print(f"  Pourcentage de valeurs manquantes : {nan_percentage:.2f}%")
    elif isinstance(res, int):  # Méthode MCMC (tau seulement)
        print(f"  Temps de changement estimé (tau) : {res}")

# === Tracé des résultats ===
plt.figure(figsize=(12, 6))
plt.plot(years, observations, label="Observations", marker="o", linestyle="None", color="black")

for name, res in results.items():
    tau_est = res[0] if isinstance(res, tuple) else res
    estimated_sizes = [res[2] if t < tau_est else res[4] for t in range(T)] if isinstance(res, tuple) else []
    if estimated_sizes:
        plt.plot(years, np.flip(estimated_sizes), linestyle="--", label=f"Taille inférée ({name})")

for change_year in theta_change_years:
    if change_year in years:
        plt.axvline(change_year, color="blue", linestyle=":", label=f"Changement de θ ({change_year})")

plt.xlabel("Années")
plt.ylabel("Nombre d'observations / Taille estimée")
plt.title(f"Estimation des tailles ({file_path})")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
