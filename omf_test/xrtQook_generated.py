# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-08-22"

Created with xrtQook




"""

import numpy as np
import sys
sys.path.append(r"D:\GitHub\xrt")
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

    beamLine.geometricSource = rsources.GeometricSource(
        bl=beamLine,
        name=None,
        center=[0, 0, 0],
        dx=0.0212,
        dz=0.0212,
        dxprime=0.002,
        dzprime=0.002)

    beamLine.Mirror = roes.EllipticalMirrorParam(
        bl=beamLine,
        name=None,
        center=[0, 5000, 0],
        pitch=r"30 mrad",
        limPhysX=[-10.0, 10.0],
        limPhysY=[-450.0, 450.0],
        p=5000)

    beamLine.screen = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 6000, 0])

    return beamLine


def run_process(beamLine):
    geometricSource01beamGlobal01 = beamLine.geometricSource.shine()

    ellipticalMirrorParam01beamGlobal01, ellipticalMirrorParam01beamLocal01 = beamLine.Mirror.reflect(
        beam=geometricSource01beamGlobal01)

    screen01beamLocal01 = beamLine.screen.expose(
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

    Source = xrtplot.XYCPlot(
        beam=r"geometricSource01beamGlobal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.4f"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            fwhmFormatStr=r"%.4f"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Source")
    plots.append(Source)

    Focus = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.4f"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            fwhmFormatStr=r"%.4f"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        aspect=r"auto",
        title=r"Focus")
    plots.append(Focus)

    Footprint = xrtplot.XYCPlot(
        beam=r"ellipticalMirrorParam01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Footprint")
    plots.append(Footprint)
    return plots


def main():
    beamLine = build_beamline()
    E0 = list(beamLine.geometricSource.energies)[0]
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()
