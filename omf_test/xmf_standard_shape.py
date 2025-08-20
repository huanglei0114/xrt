# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-08-20"

Created with xrtQook




"""

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


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.geometricSource01 = rsources.GeometricSource(
        bl=beamLine,
        name=None,
        center=[0, 0, 0],
        dx=1,
        dz=1,
        dzprime=0.001)

    beamLine.ellipticalMirrorParam01 = roes.EllipticalMirrorParam(
        bl=beamLine,
        center=[0, 5000, 0],
        pitch=r"2 deg",
        limPhysX=[-10.0, 10.0],
        p=5000)

    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        center=[0, 6000, 0])

    return beamLine


def run_process(beamLine):
    geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()

    ellipticalMirrorParam01beamGlobal01, ellipticalMirrorParam01beamLocal01 = beamLine.ellipticalMirrorParam01.reflect(
        beam=geometricSource01beamGlobal01)

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=ellipticalMirrorParam01beamGlobal01)

    outDict = {
        'geometricSource01beamGlobal01': geometricSource01beamGlobal01,
        'ellipticalMirrorParam01beamGlobal01': ellipticalMirrorParam01beamGlobal01,
        'ellipticalMirrorParam01beamLocal01': ellipticalMirrorParam01beamLocal01,
        'screen01beamLocal01': screen01beamLocal01}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"plot01")
    plots.append(plot01)
    return plots


def main():
    beamLine = build_beamline()
    E0 = list(beamLine.geometricSource01.energies)[0]
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()
