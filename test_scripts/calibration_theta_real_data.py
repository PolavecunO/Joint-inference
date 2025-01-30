# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:07:36 2025

@author: Pol
"""

import matplotlib.pyplot as plt
import numpy as np
from infer_abundance import infer_abundance
from process_yearly_data import process_yearly_data, invert_time_data

# === Paramètres ===
file_path = "data/Bombus/Bombus_(Bombus)_terrestris_(100598_specimens).csv"  # Chemin du fichier
epsilon = 1e-6
seuil_correction = 0.034
alpha_prior = 6.25
beta_prior = 0.1
theta_values = [1,0.01]  # Vecteur de valeurs de theta
theta_change_years = [1961]  # Années réelles des changements de theta

# === Extraction et traitement des données ===
try:
    years, observations = process_yearly_data(file_path)
    count_data = invert_time_data(observations)
except Exception as e:
    print(f"Erreur lors du traitement du fichier : {e}")
    exit()

# Fixer T à la longueur des données disponibles
T = len(count_data)

# Vérification des données
if T == 0:
    raise ValueError("Aucune donnée valide trouvée dans le fichier.")

# === Convertir les années réelles de changement de theta en indices ===

theta_change_times = [
        T-np.where(years == year)[0][0]  # Trouver l'index directement
        for year in theta_change_years
        if year in years
    ]


# === Inférence ===
y = count_data
tau_est, alpha_post_N0, beta_post_N0, alpha_post_N1, beta_post_N1, nan_percentage = infer_abundance(
    y=y,
    T=T,
    epsilon=epsilon,
    seuil_correction=seuil_correction,
    alpha_prior=alpha_prior,
    beta_prior=beta_prior,
    theta_values=theta_values,
    theta_change_times=theta_change_times,
)

# Calcul des tailles de population inférées
N0_est = max((alpha_post_N0 - 1) / beta_post_N0, 0) if alpha_post_N0 > 1 else 0
N1_est = max((alpha_post_N1 - 1) / beta_post_N1, 0) if alpha_post_N1 > 1 else 0

# Calcul du temps de changement en années
tau_est_year = years[::-1][tau_est] if 0 <= tau_est < T else None

# === Résultats console ===
print(f"Nombre total d'observations (T) : {T}")
print(f"Première année : {years[0]}")
print(f"Dernière année : {years[-1]}")
print(f"Temps de changement estimé (forward) : {tau_est_year}")
print(f"Taille inférée avant le changement (N0) : {N0_est}")
print(f"Taille inférée après le changement (N1) : {N1_est}")

# === Visualisation ===
time_points = np.arange(T)
estimated_sizes = [N0_est if t < tau_est else N1_est for t in time_points]

plt.figure(figsize=(12, 6))

# Plot des observations
plt.plot(years, observations, label="Observations", marker="o", linestyle="None")

# Plot des tailles estimées
plt.plot(years, np.flip(estimated_sizes), label="Taille inférée", linestyle="--", color="orange")

# Ajout des lignes pointillées pour les changements de theta
for year in theta_change_years:
    if year in years:
        plt.axvline(x=year, color="purple", linestyle=":", label=f"Changement de θ ({year})")

# Configuration du graphique
plt.xticks(years[::max(1, len(years) // 10)])
plt.xlabel("Années")
plt.ylabel("Nombre d'observations")
plt.title(f"Taille inférée ({file_path.split('/')[-1]})")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
