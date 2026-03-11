# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import sys
import os

sys.path.append(os.path.join("..", "..", ".."))
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
    i_left = np.where(h[: i_max + 1] <= half)[0]
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
    return _fwhm_from_hist(
        np.asarray(samples, dtype=float), bins=bins, rng=range, baseline=baseline
    )


# ===========================================================

# load the XRF microscope geometry from the JSON file

# VAKB-III geometry config
script_dir = Path(__file__).resolve().parent
config_path = script_dir / "Mag=110 AKB-III Geometry Config.json"

with config_path.open("r") as f:
    config = json.load(f)

mvh_theta = config["mh_theta"]
mvh_p = config["mh_q"] * 1e3
mvh_q = config["mh_p"] * 1e3
mvh_lu = config["mh_ld"] * 1e3
mvh_ld = config["mh_lu"] * 1e3

mve_theta = config["me_theta"]
mve_p = config["me_q"] * 1e3
mve_q = config["me_p"] * 1e3
mve_lu = config["me_ld"] * 1e3
mve_ld = config["me_lu"] * 1e3

mvh_l = mvh_lu + mvh_ld
mve_l = mve_lu + mve_ld


# HAKB-I geometry config
script_dir = Path(__file__).resolve().parent
config_path = script_dir / "Mag=110 AKB-I Geometry Config.json"

with config_path.open("r") as f:
    config = json.load(f)

mhh_theta = config["mh_theta"]
mhh_p = config["mh_q"] * 1e3
mhh_q = config["mh_p"] * 1e3
mhh_lu = config["mh_ld"] * 1e3
mhh_ld = config["mh_lu"] * 1e3

mhe_theta = config["me_theta"]
mhe_p = config["me_q"] * 1e3
mhe_q = config["me_p"] * 1e3
mhe_lu = config["me_ld"] * 1e3
mhe_ld = config["me_lu"] * 1e3

mhh_l = mhh_lu + mhh_ld
mhe_l = mhe_lu + mhe_ld

# ===========================================================

src_dx = 1.0e-3 / 2.355  # calculate RMS from FWHM
src_dz = 1.0e-3 / 2.355  # calculate RMS from FWHM

src_dxprime = 50e-3 / 2.355  # calculate RMS from FWHM
src_dzprime = 50e-3 / 2.355  # calculate RMS from FWHM

source_x0 = 0
source_y0 = 0
source_z0 = 0

beam_angle_x = 0
beam_angle_z = 0

mhh_x = 0
mhh_y = mhh_p
mhh_z = 0

beam_angle_x += mhh_theta * 2
beam_angle_z += 0

# screen_x = mhh_x + mhh_p * np.sin(beam_angle_x)
# screen_y = mhh_y + mhh_p * np.cos(beam_angle_x)
# screen_z = mhh_z + 0

dist = abs(mhe_p - mhh_q)

mhe_x = mhh_x + dist * np.sin(beam_angle_x) 
mhe_y = mhh_y + dist * np.cos(beam_angle_x) 
mhe_z = mhh_z + 0

beam_angle_x += mhe_theta * 2
beam_angle_z += 0

# screen_x = mhe_x + mhe_q * np.sin(beam_angle_x)
# screen_y = mhe_y + mhe_q * np.cos(beam_angle_x)
# screen_z = mhe_z + 0

dist = abs(mve_p - dist - mhh_p)

mve_x = mhe_x + dist * np.sin(beam_angle_x) 
mve_y = mhe_y + dist * np.cos(beam_angle_x) 
mve_z = mhe_z + 0

beam_angle_x += 0
beam_angle_z += mve_theta * 2

# screen_x = mve_x + (mve_q * np.cos(beam_angle_z)) * np.sin(beam_angle_x)
# screen_y = mve_y + (mve_q * np.cos(beam_angle_z)) * np.cos(beam_angle_x)
# screen_z = mve_z + mve_q * np.sin(beam_angle_z)

dist = abs(mve_q - mvh_p)

mvh_x = mve_x + (dist * np.cos(beam_angle_z)) * np.sin(beam_angle_x)
mvh_y = mve_y + (dist * np.cos(beam_angle_z)) * np.cos(beam_angle_x)
mvh_z = mve_z + dist * np.sin(beam_angle_z)

beam_angle_x += 0
beam_angle_z += - mvh_theta * 2

screen_x = mvh_x + (mvh_q * np.cos(beam_angle_z)) * np.sin(beam_angle_x)
screen_y = mvh_y + (mvh_q * np.cos(beam_angle_z)) * np.cos(beam_angle_x)
screen_z = mvh_z + mvh_q * np.sin(beam_angle_z)


def build_beamline(field_x=0e-3, field_z=0e-3): 

    source_x = source_x0  + field_x
    source_y = source_y0 
    source_z = source_z0  + field_z

    # ===========================================================

    beamLine = raycing.BeamLine()

    beamLine.geometricSource = rsources.GeometricSource(
        bl=beamLine,
        name="GS",
        center=[source_x, source_y, source_z],
        pitch=0,
        nrays=1_000_000,
        dx=src_dx,
        dz=src_dz,
        dxprime=src_dxprime,
        dzprime=src_dzprime,
        energies=[15_000.0],  # eV
    )

    beamLine.m1_mask = rapts.RectangularAperture(
        bl=beamLine,
        name="M1 Mask",
        center=[mhh_x, mhh_y, mhh_z],
        opening=[-45e-3, 40e-3, -10, 10],
        x=[1.0, 0.0, 0.0],
        z=[0.0, 0.0, 1.0],
    )
    
    beamLine.mhh = roes.ConcaveHyperbolicCylindricalMirrorXMF(
        bl=beamLine,
        name="MHH",
        center=[mhh_x, mhh_y, mhh_z],
        theta=mhh_theta,
        extraPitch=mhh_theta,
        extraRoll=np.pi/2,
        limPhysX=[-10.0, 10.0],
        limPhysY=[-mhh_lu, mhh_ld],
        p=mhh_p,
        q=mhh_q,
    )
    
    beamLine.mhe = roes.ConcaveEllipticCylindricalMirrorXMF(
        bl=beamLine,
        name="MHE",
        center=[mhe_x, mhe_y, mhe_z],
        theta=mhe_theta,
        extraPitch=mhh_theta*2+mhe_theta,
        extraRoll=np.pi/2,
        limPhysX=[-10.0, 10.0],
        limPhysY=[-mhe_lu, mhe_ld],
        p=mhe_p,
        q=mhe_q,
    )
    
    beamLine.mve = roes.ConcaveEllipticCylindricalMirrorXMF(
        bl=beamLine,
        name="MVE",
        center=[mve_x, mve_y, mve_z],
        theta=mve_theta,
        extraYaw=-(mhh_theta*2+mhe_theta*2),
        extraPitch=mve_theta,
        limPhysX=[-10.0, 10.0],
        limPhysY=[-mve_lu, mve_ld],
        p=mve_p,
        q=mve_q,
    )

    beamLine.mvh = roes.ConvexHyperbolicCylindricalMirrorXMF(
        bl=beamLine,
        name="MVH",
        center=[mvh_x, mvh_y, mvh_z],
        theta=mvh_theta,
        extraYaw=-(mhh_theta*2+mhe_theta*2),
        extraPitch=(-mve_theta*2 + mvh_theta),
        extraRoll=np.pi,
        limPhysX=[-10.0, 10.0],
        limPhysY=[-mvh_lu, mvh_ld],
        p=mvh_p,
        q=mvh_q,
    )
    

    beamLine.screen_mask = rapts.RectangularAperture(
        bl=beamLine,
        name="Screen Mask",
        center=[screen_x, screen_y, screen_z],
        opening=[-2, 2, -2, 2],
        x=[1.0, 0.0, 0.0],
        z=[0.0, 0.0, 1.0],
    )

    beamLine.screen = rscreens.Screen(
        bl=beamLine,
        name="Screen",
        center=[screen_x, screen_y, screen_z],
        x='auto',
    )

    return beamLine


def run_process(beamLine):

    geometricSource01beamGlobal01 = beamLine.geometricSource.shine()
    
    m1_mask_local = beamLine.m1_mask.propagate(beam=geometricSource01beamGlobal01)

    mhhParam01beamGlobal01, mhhParam01beamLocal01 = beamLine.mhh.reflect(
        beam=geometricSource01beamGlobal01
    )
    
    mheParam01beamGlobal01, mheParam01beamLocal01 = beamLine.mhe.reflect(
        beam=mhhParam01beamGlobal01
    )
    
    mveParam01beamGlobal01, mveParam01beamLocal01 = beamLine.mve.reflect(
        beam=mheParam01beamGlobal01
    )
    
    mvhParam01beamGlobal01, mvhParam01beamLocal01 = beamLine.mvh.reflect(
        beam=mveParam01beamGlobal01
    )
    
    screen_mask_local = beamLine.screen_mask.propagate(beam=mvhParam01beamGlobal01)
    screen01beamLocal01 = beamLine.screen.expose(beam=mvhParam01beamGlobal01)

    outDict = {
        "geometricSource01beamGlobal01": geometricSource01beamGlobal01,
        
        "m1_mask_local": m1_mask_local,
        
        "mhhParam01beamGlobal01": mhhParam01beamGlobal01,
        "mhhParam01beamLocal01": mhhParam01beamLocal01,
        
        "mheParam01beamGlobal01": mheParam01beamGlobal01,
        "mheParam01beamLocal01": mheParam01beamLocal01,

        "mveParam01beamGlobal01": mveParam01beamGlobal01,
        "mveParam01beamLocal01": mveParam01beamLocal01,
        
        "mvhParam01beamGlobal01": mvhParam01beamGlobal01,
        "mvhParam01beamLocal01": mvhParam01beamLocal01,
        
        "screen_mask_local": screen_mask_local,
        "screen01beamLocal01": screen01beamLocal01,
    }

    beamLine.prepare_flow()

    # === FWHM at screen (local coordinates) ==============================
    # Keep only good rays
    b = screen01beamLocal01
    # XRT typically flags good rays with state == 1
    good = b.state == 1

    x = b.x[good]  # meters
    z = b.z[good]  # meters
    xr = (np.nanmin(x), np.nanmax(x))
    zr = (np.nanmin(z), np.nanmax(z))

    fwhm_x, xL, xR = fwhm_from_samples(
        x, bins=min([round(np.sum(good) / 100), 512]), range=xr, baseline=0.0
    )
    fwhm_z, zL, zR = fwhm_from_samples(
        z, bins=min([round(np.sum(good) / 100), 512]), range=zr, baseline=0.0
    )

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
        beam=r"geometricSource01beamGlobal01",
        xaxis=xrtplot.XYCAxis(label=r"x", fwhmFormatStr=r"%.3f", unit="um", factor=1e3),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            fwhmFormatStr=r"%.3f",
            # limits=[source_z0*1e3-1500, source_z0*1e3+1500],
            # offset=source_z0*1e3,
            unit="um",
            factor=1e3,
        ),
        caxis=xrtplot.XYCAxis(label=r"energy", unit=r"eV"),
        title=r"Source",
        aspect="equal",
    )
    plots.append(Source)

    MHH_Footprint = xrtplot.XYCPlot(
        beam=r"mhhParam01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f",
            # limits=[-1_000, 1_000],
            unit="um",
            factor=1e3,
        ),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
            fwhmFormatStr=r"%.3f",
            limits=[-mvh_lu * 1e3, mvh_ld * 1e3],
            unit="um",
            factor=1e3,
        ),
        caxis=xrtplot.XYCAxis(label=r"energy", unit=r"eV"),
        title=r"Footprint",
        aspect="auto",
    )
    plots.append(MHH_Footprint)
    
    MHE_Footprint = xrtplot.XYCPlot(
        beam=r"mheParam01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f",
            # limits=[-5_000, 5_000],
            unit="um",
            factor=1e3,
        ),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
            fwhmFormatStr=r"%.3f",
            limits=[-mve_lu * 1e3, mve_ld * 1e3],
            unit="um",
            factor=1e3,
        ),
        caxis=xrtplot.XYCAxis(label=r"energy", unit=r"eV"),
        title=r"Footprint",
        aspect="auto",
    )
    plots.append(MHE_Footprint)
    
    MVE_Footprint = xrtplot.XYCPlot(
        beam=r"mveParam01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f",
            # limits=[-5_000, 5_000],
            unit="um",
            factor=1e3,
        ),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
            fwhmFormatStr=r"%.3f",
            limits=[-mve_lu * 1e3, mve_ld * 1e3],
            unit="um",
            factor=1e3,
        ),
        caxis=xrtplot.XYCAxis(label=r"energy", unit=r"eV"),
        title=r"Footprint",
        aspect="auto",
    )
    plots.append(MVE_Footprint)
    
    MVH_Footprint = xrtplot.XYCPlot(
        beam=r"mvhParam01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f",
            # limits=[-5_000, 5_000],
            unit="um",
            factor=1e3,
        ),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
            fwhmFormatStr=r"%.3f",
            limits=[-mve_lu * 1e3, mve_ld * 1e3],
            unit="um",
            factor=1e3,
        ),
        caxis=xrtplot.XYCAxis(label=r"energy", unit=r"eV"),
        title=r"Footprint",
        aspect="auto",
    )
    plots.append(MVH_Footprint)   
     
    Focus = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(label=r"x", fwhmFormatStr=r"%.3f", unit="um", factor=1e3),
        yaxis=xrtplot.XYCAxis(label=r"z", fwhmFormatStr=r"%.3f", unit="um", factor=1e3),
        caxis=xrtplot.XYCAxis(label=r"energy", unit=r"eV"),
        aspect=r"auto",
        title=r"Focus",
    )
    plots.append(Focus)

    return plots


def main():

    fwhm_x_um = []
    fwhm_z_um = []
    field_x_um = np.linspace(0, 0, 1)
    field_z_um = np.linspace(0, 0, 1)
    for field_x in field_x_um * 1e-3:
        for field_z in field_z_um * 1e-3:
            beamLine = build_beamline(field_x=field_x, field_z=field_z)

            E0 = list(beamLine.geometricSource.energies)[0]
            beamLine.alignE = E0
            plots = define_plots()
            xrtrun.run_ray_tracing(
                plots=plots, 
                repeats=1,
                processes=1, 
                backend=r"raycing", 
                beamLine=beamLine
            )
            beamLine.glow()
            fwhm_x_um.append(beamLine.fwhm_x * 1e3)
            fwhm_z_um.append(beamLine.fwhm_z * 1e3)

    return fwhm_x_um, fwhm_z_um

if __name__ == "__main__":
    main()
