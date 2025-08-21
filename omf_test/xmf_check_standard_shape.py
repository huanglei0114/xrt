# -*- coding: utf-8 -*-
"""

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
        dzprime=0.001)

    beamLine.mirror = roes.EllipsoidalMirrorXMF(
        bl=beamLine,
        name=None,
        center=[0, 5000, 0],
        pitch=r"2 deg",
        limPhysX=[-10.0, 10.0],
        p=5000,
        q=1000)
    
    # beamLine.mirror = roes.EllipticalMirror(
    #     bl=beamLine,
    #     name=None,
    #     center=[0, 5000, 0],
    #     pitch=r"2 deg",
    #     limPhysX=[-10.0, 10.0],
    #     p=5000,
    #     q=1000)
    
    beamLine.screen = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 5999.3908, 34.8995])

    return beamLine


def run_process(beamLine):
    geometricSource01beamGlobal01 = beamLine.geometricSource.shine()

    mirror01beamGlobal01, mirror01beamLocal01 = beamLine.mirror.reflect(
        beam=geometricSource01beamGlobal01)

    screen01beamLocal01 = beamLine.screen.expose(
        beam=mirror01beamGlobal01)

    outDict = {
        'geometricSource01beamGlobal01': geometricSource01beamGlobal01,
        'mirror01beamGlobal01': mirror01beamGlobal01,
        'mirror01beamLocal01': mirror01beamLocal01,
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
