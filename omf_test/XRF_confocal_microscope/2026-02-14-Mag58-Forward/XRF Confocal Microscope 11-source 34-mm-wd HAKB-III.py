# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

sys.path.append(os.path.join('..', '..', '..'))
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

me_theta = 4.5e-3
me_p = 53.0510
me_q = 211.8167
me_lu = 18.9584
me_ld = 81.4451
me_l = me_lu + me_ld

mh_theta = 0.8513e-3
mh_p = 72.5576
mh_q = 1307.6925
mh_lu = 55.6925
mh_ld = 55.6925
mh_l = mh_lu + mh_ld

src_dx = 0.5e-3/2.355 # calculate RMS from FWHM
src_dz = 0.5e-3/2.355 # calculate RMS from FWHM

src_dxprime = 100e-3/2.355 # calculate RMS from FWHM
src_dzprime = 100e-3/2.355 # calculate RMS from FWHM

source_y0 = - me_p * np.cos(me_theta)
source_z0 = me_p * np.sin(me_theta)

mh_y = abs(me_q - mh_p) * np.cos(me_theta)
mh_z = abs(me_q - mh_p) * np.sin(me_theta)

scr_y = mh_y + mh_q * np.cos(me_theta - 2*mh_theta)
scr_z = mh_z + mh_q * np.sin(me_theta - 2*mh_theta)

field_z1d = np.linspace(-10e-3, 10e-3, 11) # field size in z direction

def build_beamline(nrays_per_source=1_000_000): # field size in z direction

    beamLine = raycing.BeamLine()
    beamLine.gs = {}
    for idx, field_z in enumerate(field_z1d):
        source_y = source_y0 + field_z * np.sin(me_theta)
        source_z = source_z0 + field_z * np.cos(me_theta)
        name = f"GS{idx:02d}"
        beamLine.gs[name] = rsources.GeometricSource(
            bl=beamLine,
            name=name,
            center=[0, source_y, source_z],
            pitch=-me_theta,
            nrays=nrays_per_source,
            dx=src_dx,
            dz=src_dz,
            dxprime=src_dxprime,
            dzprime=src_dzprime)

    beamLine.me = roes.ConcaveEllipticCylindricalMirrorXMF(
        bl=beamLine,
        name="EM",
        center=[0, 0, 0],
        theta=me_theta,
        limPhysX=[-10.0, 10.0],
        limPhysY=[-me_lu, me_ld],
        p=me_p,
        q=me_q,
        )

    beamLine.mh = roes.ConvexHyperbolicCylindricalMirrorXMF(
        bl=beamLine,
        name="HM",
        center=[0, mh_y, mh_z],
        theta=mh_theta,        
        extraPitch = (- me_theta + mh_theta),
        extraRoll = np.pi,
        limPhysX=[-10.0, 10.0],
        limPhysY=[-mh_lu, mh_ld],
        p=mh_p,
        q=mh_q,
        )
    
    beamLine.screen_mask = rapts.RectangularAperture(
        bl=beamLine,
        name="Mask",
        center=[0, scr_y, scr_z],
        opening=[-1e3, 1e3, -1000e-3, 1000e-3],
        x=[1.0, 0.0, 0.0],
        z=[0.0, 0.0, 1.0])
    
    beamLine.screen = rscreens.Screen(
        bl=beamLine,
        name="SCR",
        center=[0, scr_y, scr_z],
        z=[0, -np.sin(me_theta - 2*mh_theta), np.cos(me_theta - 2*mh_theta)]
    )

    return beamLine


def run_process(beamLine):
    
    # Make an empty beam container and append each source beam into it
    for idx, field_z in enumerate(field_z1d):
        name = f"GS{idx:02d}"
        src_beam = beamLine.gs[name].shine()
        if idx == 0:
            beam_total = src_beam
        else:
            beam_total.concatenate(src_beam)

    meParam01beamGlobal01, meParam01beamLocal01 = beamLine.me.reflect(
        beam=beam_total)
    
    mhParam01beamGlobal01, mhParam01beamLocal01 = beamLine.mh.reflect(
        beam=meParam01beamGlobal01)

    screen_mask_local = beamLine.screen_mask.propagate(
        beam=mhParam01beamGlobal01)
    
    screen01beamLocal01 = beamLine.screen.expose(
        beam=mhParam01beamGlobal01)

    outDict = {
        'beam_total': beam_total,
        
        'meParam01beamGlobal01': meParam01beamGlobal01,
        'meParam01beamLocal01': meParam01beamLocal01,
        
        'mhParam01beamGlobal01': mhParam01beamGlobal01,
        'mhParam01beamLocal01': mhParam01beamLocal01,

        'screen_mask_local': screen_mask_local,
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

    fwhm_x, xL, xR = fwhm_from_samples(x, bins=min([round(np.sum(good)/100), 512]), range=xr, baseline=0.0)
    fwhm_z, zL, zR = fwhm_from_samples(z, bins=min([round(np.sum(good)/100), 512]), range=zr, baseline=0.0)

    print(f"[Screen @ local]  FWHM_x = {fwhm_x:.6e} mm  ({fwhm_x*1e3:.3f} µm)")
    print(f"[Screen @ local]  FWHM_z = {fwhm_z:.6e} mm  ({fwhm_z*1e3:.3f} µm)")
    # =====================================================================
    
    beamLine.fwhm_x = fwhm_x
    beamLine.fwhm_z = fwhm_z
    beamLine.screen01beamLocal01 = screen01beamLocal01

    return outDict

rrun.run_process = run_process



def define_plots():
    plots = []

    Source = xrtplot.XYCPlot(
        beam=r"beam_total",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f",
            unit="um",
            factor=1e3),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            fwhmFormatStr=r"%.3f",
            # limits=[source_z0*1e3-1500, source_z0*1e3+1500],
            # offset=source_z0*1e3,
            unit="um",
            factor=1e3),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Source",
        aspect="equal")
    plots.append(Source)
    
    ME_Footprint = xrtplot.XYCPlot(
        beam=r"meParam01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f",
            # limits=[-5_000, 5_000],
            unit="um",
            factor=1e3),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
            fwhmFormatStr=r"%.3f",
            limits=[-me_lu*1e3, me_ld*1e3],
            unit="um",
            factor=1e3),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Footprint",
        aspect="auto")
    plots.append(ME_Footprint)

    MH_Footprint = xrtplot.XYCPlot(
        beam=r"mhParam01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f",
            # limits=[-1_000, 1_000],
            unit="um",
            factor=1e3),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
            fwhmFormatStr=r"%.3f",
            limits=[-mh_lu*1e3, mh_ld*1e3],
            unit="um",
            factor=1e3),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Footprint",
        aspect="auto")
    plots.append(MH_Footprint)
    
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
    
    beamLine = build_beamline()
        
    E0 = list(beamLine.gs['GS02'].energies)[0]
    beamLine.alignE = E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=1,
        processes=1,
        backend=r"raycing",
        beamLine=beamLine)
    beamLine.glow()

    # === FWHM at screen (local coordinates) ==============================

    # Keep only good rays
    b = beamLine.screen01beamLocal01
    # XRT typically flags good rays with state == 1
    good = b.state == 1
    print(f"[good]  good = {np.sum(good)}")
    
    x = b.x[good]   # meters
    z = b.z[good]   # meters
    xr = (np.nanmin(x), np.nanmax(x))
    zr = (np.nanmin(z), np.nanmax(z))

    fwhm_x, xL, xR = fwhm_from_samples(x, bins=min(round(np.sum(good)/100), 512), range=xr, baseline=0.0)
    fwhm_z, zL, zR = fwhm_from_samples(z, bins=min([round(np.sum(good)/100), 512]), range=zr, baseline=0.0)

    print(f"[Screen @ local]  FWHM_x = {fwhm_x:.6e} mm  ({fwhm_x*1e3:.3f} µm)")
    print(f"[Screen @ local]  FWHM_z = {fwhm_z:.6e} mm  ({fwhm_z*1e3:.3f} µm)")
    
    # Combined figure: scatter (x, z) and histogram of z
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot of (x, z)
    ax[0].scatter(x*1e3, z*1e3, s=1)
    ax[0].set_xlabel("x (µm)")
    ax[0].set_ylabel("z (µm)")
    ax[0].set_title("Scatter plot of (x, z) at the screen")
    ax[0].grid()

    # Histogram of z
    ax[1].hist(z*1e3, bins=100, range=(zr[0]*1e3, zr[1]*1e3))
    ax[1].set_xlabel("z (µm)")
    ax[1].set_ylabel("Counts")
    ax[1].set_title("Histogram of z at the screen")
    ax[1].grid()

    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    main()
