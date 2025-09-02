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

import xrtCodeForLei as roes_joe


m1_theta = 30e-3
m1_p = 5000
m1_q = 1000
source_y = - m1_p * np.cos(m1_theta)
source_z = m1_p * np.sin(m1_theta)

scr_y = m1_q * np.cos(m1_theta)
scr_z = m1_q * np.sin(m1_theta)

src_dx = 212e-6
src_dz = 212e-6

src_dxprime = 2e-3
src_dzprime = 2e-3

def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.geometricSource = rsources.GeometricSource(
        bl=beamLine,
        name="GS",
        center=[0, source_y, source_z],
        pitch=-m1_theta,
        dx=src_dx,
        dz=src_dz,
        dxprime=src_dxprime,
        dzprime=src_dzprime)
    
    beamLine.mirror_xmf = roes.TanColDiaboloidalMirrorXMF(
        bl=beamLine,
        name=None,
        center=[0, 0, 0],
        pitch=m1_theta,
        extraPitch=-m1_theta,
        limPhysX=[-10.0, 10.0],
        limPhysY=[-500.0, 500.0],
        p=m1_p,
        q=m1_q
        )
    
    beamLine.mirror_joe = roes_joe.TanCollDiaboloidMirrorJoe(
        bl=beamLine,
        name=None,
        center=[0, 0, 0],
        theta=m1_theta,
        #extraPitch=-m1_theta,
        limPhysX=[-10.0, 10.0],
        limPhysY=[-500.0, 500.0],
        p=m1_p,
        q=m1_q
        )
 
    beamLine.mirror = beamLine.mirror_xmf # Using XMF TCD
    # beamLine.mirror = beamLine.mirror_joe # Using Joe's TCD
    
    beamLine.screen = rscreens.Screen(
        bl=beamLine,
        name="SCR",
        center=[0, scr_y, scr_z],
        z=[0, -np.sin(m1_theta), np.cos(m1_theta)])

    return beamLine


def run_process(beamLine):
    geometricSource01beamGlobal01 = beamLine.geometricSource.shine()

    M1Param01beamGlobal01, M1Param01beamLocal01 = beamLine.mirror.reflect(
        beam=geometricSource01beamGlobal01)

    screen01beamLocal01 = beamLine.screen.expose(
        beam=M1Param01beamGlobal01)
   
    outDict = {
        'geometricSource01beamGlobal01': geometricSource01beamGlobal01,
        'M1Param01beamGlobal01': M1Param01beamGlobal01,
        'M1Param01beamLocal01': M1Param01beamLocal01,
        'screen01beamLocal01': screen01beamLocal01}
    beamLine.prepare_flow()
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    Source = xrtplot.XYCPlot(
        beam=r"geometricSource01beamGlobal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.6f",
            limits=[-1e-3, 1e-3],
            unit="mm",
            factor=1),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            fwhmFormatStr=r"%.6f",
            limits=[source_z-1e-3, source_z+1e-3],
            offset=source_z,
            unit="mm",
            factor=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Source",
        aspect="equal")
    plots.append(Source)

    Footprint = xrtplot.XYCPlot(
        beam=r"M1Param01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f"),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
            fwhmFormatStr=r"%.3f"),
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
            fwhmFormatStr=r"%.1f",
            # limits=[-200, 200],
            unit="nm",
            factor=1e6),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            fwhmFormatStr=r"%.1f",
            # limits=[-200, 200],
            unit="mm",
            factor=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        aspect=r"auto",
        title=r"Focus")
    plots.append(Focus)
    
    return plots


def main():
    beamLine = build_beamline()
    # beamLine.glow()
    E0 = list(beamLine.geometricSource.energies)[0]
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=16,
        processes=4,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()
