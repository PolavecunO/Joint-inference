# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:21:56 2025

@author: Pol
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_error_heatmaps(results_matrix, meta_info, num_per_window=10):
    binary_param_values = meta_info["variations"]["binary_param"]
    param_values = meta_info["variations"]["param_values"]
    N0_values = meta_info["variations"]["N0"]
    N1_values = meta_info["variations"]["N1"]

    for b_idx, binary_value in enumerate(binary_param_values):
        fixed_param_split = np.array_split(param_values, len(param_values) // num_per_window + (len(param_values) % num_per_window > 0))

        for window_idx, subset in enumerate(fixed_param_split):
            fig, axes = plt.subplots(2, 5, figsize=(24, 12))
            fig.suptitle(f"Erreur moyenne pour {binary_value} (Fenêtre {window_idx + 1})", fontsize=16)

            for idx, param_value in enumerate(subset):
                heatmap_data = results_matrix[b_idx, param_values.index(param_value), :, :, 1:3]
                heatmap_data = np.mean(np.abs(heatmap_data[..., 0] - heatmap_data[..., 1]) / 
                                       (heatmap_data[..., 0] + heatmap_data[..., 1]), axis=-1)

                # Vérifier si heatmap_data est bien 2D
                if heatmap_data.ndim != 2:
                    raise ValueError(f"heatmap_data has invalid shape {heatmap_data.shape}. Expected 2D.")

                row, col = divmod(idx, 5)
                ax = axes[row, col]

                im = ax.imshow(
                    heatmap_data,
                    cmap="viridis",
                    origin="lower",
                    extent=[min(N0_values), max(N0_values), min(N1_values), max(N1_values)],
                    aspect="auto"
                )
                ax.set_title(f"Paramètre = {param_value}")
                ax.set_xlabel("N0")
                ax.set_ylabel("N1")
                fig.colorbar(im, ax=ax, label="Erreur moyenne")

            for idx in range(len(subset), 10):
                row, col = divmod(idx, 5)
                fig.delaxes(axes[row, col])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

def plot_error_difference_heatmaps(results_matrix_1, results_matrix_2, meta_info, num_per_window=10):
    """
    Affiche des heatmaps des différences d'erreurs moyennes entre deux matrices, 
    pour chaque valeur du paramètre quantitatif.

    Arguments :
    - results_matrix_1, results_matrix_2 : Deux matrices multidimensionnelles des résultats.
    - meta_info : Dictionnaire contenant les métadonnées.
    - num_per_window : Nombre de heatmaps par fenêtre (par défaut 10).
    """
    param_values = meta_info["variations"][meta_info["dimensions"][1]]
    N0_values = meta_info["variations"]["N0"]
    N1_values = meta_info["variations"]["N1"]

    fixed_param_split = np.array_split(param_values, len(param_values) // num_per_window + (len(param_values) % num_per_window > 0))

    for window_idx, subset in enumerate(fixed_param_split):
        fig, axes = plt.subplots(2, 5, figsize=(24, 12))
        fig.suptitle(f"Différence d'erreurs moyennes (Fenêtre {window_idx + 1})", fontsize=16)

        for idx, param_value in enumerate(subset):
            param_idx = param_values.index(param_value)

            # Calculer la différence d'erreur moyenne pour chaque combinaison de N0 et N1
            error_diff = (
                np.mean(results_matrix_1[:, param_idx, ..., 1] - results_matrix_1[:, param_idx, ..., 2], axis=0) -
                np.mean(results_matrix_2[:, param_idx, ..., 1] - results_matrix_2[:, param_idx, ..., 2], axis=0)
            )

            row, col = divmod(idx, 5)
            ax = axes[row, col]

            im = ax.imshow(
                error_diff,
                cmap="coolwarm",
                origin="lower",
                extent=[min(N0_values), max(N0_values), min(N1_values), max(N1_values)],
                aspect="auto"
            )
            ax.set_title(f"{meta_info['dimensions'][1]} = {param_value}")
            ax.text(0.5, -0.15, f"Diff. Moy.={np.mean(error_diff):.4f}", transform=ax.transAxes, fontsize=10, ha='center')
            ax.set_xlabel("N0")
            if col == 0:
                ax.set_ylabel("N1")
            fig.colorbar(im, ax=ax, label="Différence d'erreurs moyennes")

        for idx in range(len(subset), num_per_window):
            row, col = divmod(idx, 5)
            fig.delaxes(axes[row, col])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def plot_error_vs_param(results_matrix, meta_info):
    """
    Trace l'erreur moyenne sur N0 et N1 en fonction du paramètre quantitatif, 
    avec une courbe par valeur du paramètre binaire.

    Arguments :
    - results_matrix : Matrice multidimensionnelle des résultats.
    - meta_info : Dictionnaire contenant les métadonnées.
    """
    binary_values = meta_info["variations"][meta_info["dimensions"][0]]
    param_values = meta_info["variations"][meta_info["dimensions"][1]]

    plt.figure(figsize=(12, 6))

    for binary_idx, binary_value in enumerate(binary_values):
        # Moyenne sur toutes les combinaisons de N0 et N1
        mean_errors = np.mean(
            np.abs(results_matrix[binary_idx, ..., 1] - results_matrix[binary_idx, ..., 2]) * 2 /
            (results_matrix[binary_idx, ..., 1] + results_matrix[binary_idx, ..., 2]),
            axis=(1, 2)
        )

        plt.plot(param_values, mean_errors, label=f"{meta_info['dimensions'][0]} = {binary_value}")

    plt.xlabel(meta_info["dimensions"][1])
    plt.ylabel("Erreur moyenne")
    plt.title("Erreur moyenne en fonction du paramètre quantitatif")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
