# -*- coding: utf-8 -*-

import numpy as np
import sys
sys.path.append(r"/Users/lhuang/Documents/GitHub/xrt")
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.materials_elemental as rmatsel
import xrt.backends.raycing.materials_compounds as rmatsco
import xrt.backends.raycing.materials_crystals as rmatscr
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

import matplotlib.pyplot as plt

def _fwhm_from_hist(x, bins=201, rng=None, baseline=0.0):
    """
    Compute FWHM from 1D samples by histogramming and linearly
    interpolating the half-max crossings. Returns (width, xL, xR).
    """
    h, edges = np.histogram(x, bins=bins, range=rng)
    c = 0.5 * (edges[:-1] + edges[1:])

    # Peak & half level (with baseline)
    i_max = np.argmax(h)
    y_max = h[i_max]
    half = baseline + 0.5 * (y_max - baseline)

    # Left side crossing
    i_left = np.where(h[:i_max+1] <= half)[0]
    if i_left.size == 0:
        xL = c[0]
    else:
        i1 = i_left[-1]
        i2 = min(i1 + 1, i_max)
        xL = c[i1] + (half - h[i1]) * (c[i2] - c[i1]) / (h[i2] - h[i1] + 1e-30)

    # Right side crossing
    seg = h[i_max:]
    i_right = np.where(seg <= half)[0]
    if i_right.size == 0:
        xR = c[-1]
    else:
        j2 = i_right[0] + i_max
        j1 = max(j2 - 1, i_max)
        xR = c[j1] + (half - h[j1]) * (c[j2] - c[j1]) / (h[j2] - h[j1] + 1e-30)

    return (xR - xL), xL, xR


def fwhm_from_samples(samples, bins=201, range=None, baseline=0.0):
    """Tiny convenience wrapper."""
    return _fwhm_from_hist(np.asarray(samples, dtype=float),
                           bins=bins, rng=range, baseline=baseline)
# ===========================================================

m1_theta = 3.4985e-3
m1_p = 1_494.8531
m1_q = 5.1476

src_dx = 145.2e-3/2.355 # calculate RMS from FWHM
src_dz = 145.2e-3/2.355 # calculate RMS from FWHM

src_dxprime = 1e-3/2.355 # calculate RMS from FWHM
src_dzprime = 1e-3/2.355 # calculate RMS from FWHM

source_y0 = - m1_p * np.cos(m1_theta)
source_z0 = m1_p * np.sin(m1_theta)

scr_y = m1_q * np.cos(m1_theta)
scr_z = m1_q * np.sin(m1_theta)


    
def build_beamline(field_z = 0e-3): # field size in z direction

    source_y = source_y0 + field_z * np.sin(m1_theta)
    source_z = source_z0 + field_z * np.cos(m1_theta)
    
    # ===========================================================
    
    beamLine = raycing.BeamLine()

    
    
    
    beamLine.geometricSource = rsources.GeometricSource(
        bl=beamLine,
        name="GS",
        center=[0, source_y, source_z],
        pitch=-m1_theta,
        dx=src_dx,
        dz=src_dz,
        nrays=500000,  # increase the number of rays
        dxprime=src_dxprime,
        dzprime=src_dzprime)

    beamLine.mirror = roes.EllipticalMirror(
        bl=beamLine,
        name="EM",
        center=[0, 0, 0],
        pitch=m1_theta,
        extraPitch=-m1_theta,
        limPhysX=[-10.0, 10.0],
        limPhysY=[-10.0518, 3.3506],
        p=m1_p,
        q=m1_q,
        isCylindrical=True
        )

    beamLine.screen = rscreens.Screen(
        bl=beamLine,
        name="SCR",
        center=[0, scr_y, scr_z],
        z=[0, -np.sin(m1_theta), np.cos(m1_theta)]
    )

    return beamLine


def run_process(beamLine):
    geometricSource01beamGlobal01 = beamLine.geometricSource.shine()

    mrrorParam01beamGlobal01, mrrorParam01beamLocal01 = beamLine.mirror.reflect(
        beam=geometricSource01beamGlobal01)

    screen01beamLocal01 = beamLine.screen.expose(
        beam=mrrorParam01beamGlobal01)

    outDict = {
        'geometricSource01beamGlobal01': geometricSource01beamGlobal01,
        'mrrorParam01beamGlobal01': mrrorParam01beamGlobal01,
        'mrrorParam01beamLocal01': mrrorParam01beamLocal01,
        'screen01beamLocal01': screen01beamLocal01}
    beamLine.prepare_flow()
    
    # === FWHM at screen (local coordinates) ==============================
    # Keep only good rays
    b = screen01beamLocal01
    # XRT typically flags good rays with state == 1
    good = (b.state == 1)

    x = b.x[good]   # meters
    z = b.z[good]   # meters
    xr = (np.nanmin(x), np.nanmax(x))
    zr = (np.nanmin(z), np.nanmax(z))

    fwhm_x, xL, xR = fwhm_from_samples(x, bins=round(np.sum(good)/100), range=xr, baseline=0.0)
    fwhm_z, zL, zR = fwhm_from_samples(z, bins=round(np.sum(good)/100), range=zr, baseline=0.0)

    print(f"[Screen @ local]  FWHM_x = {fwhm_x:.6e} mm  ({fwhm_x*1e3:.3f} µm)")
    print(f"[Screen @ local]  FWHM_z = {fwhm_z:.6e} mm  ({fwhm_z*1e3:.3f} µm)")
    # =====================================================================
    
    beamLine.fwhm_x = fwhm_x
    beamLine.fwhm_z = fwhm_z

    return outDict

rrun.run_process = run_process



def define_plots():
    plots = []

    Source = xrtplot.XYCPlot(
        beam=r"geometricSource01beamGlobal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f",
            unit="um",
            factor=1e3),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            fwhmFormatStr=r"%.3f",
            limits=[source_z0*1e3-500, source_z0*1e3+500],
            offset=source_z0*1e3,
            unit="um",
            factor=1e3),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Source",
        aspect="equal")
    plots.append(Source)

    Footprint = xrtplot.XYCPlot(
        beam=r"mrrorParam01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f",
            # limits=[-100, 100],
            unit="um",
            factor=1e3),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
            fwhmFormatStr=r"%.3f",
            limits=[-11_000, 4_000],
            unit="um",
            factor=1e3),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Footprint",
        aspect="auto")
    plots.append(Footprint)

    Focus = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f",
            unit="um",
            factor=1e3),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            fwhmFormatStr=r"%.3f",
            unit="um",
            factor=1e3),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        aspect=r"auto",
        title=r"Focus")
    plots.append(Focus)

    return plots



def main():
    
    fwhm_x_um = []
    fwhm_z_um = []
    field_z_um = np.linspace(-1, 1, 21) * 1e3
    for field_z in field_z_um * 1e-3:
        beamLine = build_beamline(field_z)
        E0 = list(beamLine.geometricSource.energies)[0]
        beamLine.alignE=E0
        # plots = define_plots()
        xrtrun.run_ray_tracing(
            # plots=plots,
            repeats=1,
            processes=1,
            backend=r"raycing",
            beamLine=beamLine)
        # beamLine.glow()
        fwhm_x_um.append(beamLine.fwhm_x*1e3)
        fwhm_z_um.append(beamLine.fwhm_z*1e3)

    print("Field position in z direction (µm):", field_z_um)
    print("FWHM X (µm):", fwhm_x_um)
    print("FWHM Z (µm):", fwhm_z_um)
    
    # plot FWHM vs field size
    plt.figure(figsize=(16,9))
    plt.plot(field_z_um, fwhm_z_um, '-o')
    plt.xlabel("Field position in z direction (µm)")
    plt.ylabel("FWHM in z direction (µm)")
    plt.grid()
    plt.title("FWHM in z direction vs field position in z direction")
    plt.tight_layout()
    plt.show()
    



if __name__ == '__main__':
    main()
