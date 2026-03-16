import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

# Load FWHM data from JSON file

script_dir = Path(__file__).parent
fwhm_data_path = script_dir / "fwhm_data.json"
with fwhm_data_path.open("r") as f:
    fwhm_data = json.load(f)
field_x_um = np.array(fwhm_data["field_x_um"])
field_z_um = np.array(fwhm_data["field_z_um"])
fwhm_x_um = np.array(fwhm_data["fwhm_x_um"])
fwhm_z_um = np.array(fwhm_data["fwhm_z_um"])


# Plot FWHM vs field
if field_x_um.size > 1 and field_z_um.size == 1:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(field_x_um, fwhm_z_um, "--", marker="o", ms=10, lw=2, label="FWHM_z", color="C1")
    ax.plot(field_x_um, fwhm_x_um, marker="X", ms=10, lw=2, label="FWHM_x", color="C0")
    ax.set_xlabel("x-Field [µm]", fontsize=14)
    ax.set_ylabel("FWHM [µm]", fontsize=14)
    ax.set_title("FWHM_x at Screen vs Field of View", fontsize=14)
    # ax.set_ylim([0, 100])
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure to a svg file
    output_fig_path = script_dir / "fwhm_vs_field_x.svg"
    plt.savefig(output_fig_path, dpi=600)

if field_z_um.size > 1 and field_x_um.size == 1:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(field_z_um, fwhm_x_um, "--",marker="X", ms=10, lw=2, label="FWHM_x", color="C0")
    ax.plot(field_z_um, fwhm_z_um, "-", marker="o", ms=10, lw=2, label="FWHM_z", color="C1")
    ax.set_xlabel("z-Field [µm]", fontsize=14)
    ax.set_ylabel("FWHM [µm]", fontsize=14)
    ax.set_title("FWHM at Screen vs Field of View", fontsize=14)
    # ax.set_ylim([0, 100])
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure to a svg file
    output_fig_path = script_dir / "fwhm_vs_field_z.svg"
    plt.savefig(output_fig_path, dpi=600)

if field_x_um.size > 1 and field_z_um.size > 1:

    field_x2d_um, field_z2d_um = np.meshgrid(field_x_um, field_z_um)
    fwhm_x_um = np.array(fwhm_x_um).reshape(field_z_um.size, field_x_um.size)
    fwhm_z_um = np.array(fwhm_z_um).reshape(field_z_um.size, field_x_um.size)
    fwhm_um = np.sqrt(fwhm_x_um**2 + fwhm_z_um**2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sc = axes[0].pcolormesh(
        field_x2d_um,
        field_z2d_um,
        fwhm_x_um,
        shading="auto",
        cmap="viridis",
    )
    axes[0].set_xlabel("x-Field [µm]", fontsize=14)
    axes[0].set_ylabel("z-Field [µm]", fontsize=14)
    axes[0].set_title("x-FWHM at Screen vs Field", fontsize=14)
    plt.colorbar(sc, label="FWHM [µm]")

    sc = axes[1].pcolormesh(
        field_x2d_um,
        field_z2d_um,
        fwhm_z_um,
        shading="auto",
        cmap="viridis",
    )
    axes[1].set_xlabel("x-Field [µm]", fontsize=14)
    axes[1].set_ylabel("z-Field [µm]", fontsize=14)
    axes[1].set_title("z-FWHM at Screen vs Field", fontsize=14)
    plt.colorbar(sc, label="FWHM [µm]")

    sc = axes[2].pcolormesh(
        field_x2d_um,
        field_z2d_um,
        fwhm_um,
        shading="auto",
        cmap="plasma",
    )
    axes[2].set_xlabel("x-Field [µm]", fontsize=14)
    axes[2].set_ylabel("z-Field [µm]", fontsize=14)
    axes[2].set_title("Total FWHM at Screen vs Field", fontsize=14)
    plt.colorbar(sc, label="FWHM [µm]")

    plt.tight_layout()
    
    # Save the figure to a svg file
    output_fig_path = script_dir / "fwhm_vs_field_2d.svg"
    plt.savefig(output_fig_path, dpi=600)

plt.show()
