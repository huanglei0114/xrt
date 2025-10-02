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

m1_theta = 2/180*np.pi

m1_p = 50_000
m1_q_t = -50_000
m1_q_s = 5_000
src_dxprime = 4e-4
src_dzprime = 1e-4
src_dx = 212e-6*1e-6
src_dz = 212e-6*1e-6



# m1_p = 1800
# m1_q_t = 6200
# m1_q_s = 10000
# src_dxprime = 8e-3
# src_dzprime = 2e-3
# src_dx = 212e-6*1e-6
# src_dz = 212e-6*1e-6


source_y = - m1_p * np.cos(m1_theta)
source_z = m1_p * np.sin(m1_theta)

scr_t_y = m1_q_t * np.cos(m1_theta)
scr_t_z = m1_q_t * np.sin(m1_theta)

scr_s_y = m1_q_s * np.cos(m1_theta)
scr_s_z = m1_q_s * np.sin(m1_theta)





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
    
    beamLine.mirror_xmf = roes.P1L2DiaboloidalMirrorXMF(
        bl=beamLine,
        name=None,
        center=[0, 0, 0],
        theta=m1_theta,
        limPhysX=[-25.0, 25.0],
        limPhysY=[-175.0, 175.0],
        p=m1_p,
        q_t=m1_q_t,
        q_s=m1_q_s,
        )
    
    # beamLine.mirror_ellipsoid_xmf = roes.ConcaveEllipsoidalMirrorXMF(
    #     bl=beamLine,
    #     name=None,
    #     center=[0, 0, 0],
    #     theta=m1_theta,
    #     limPhysX=[-25.0, 25.0],
    #     limPhysY=[-175.0, 175.0],
    #     p=m1_p,
    #     q=m1_q_t,
    #     )
    
    # beamLine.mirror_ellipse_xmf = roes.ConcaveEllipticCylindricalMirrorXMF(
    #     bl=beamLine,
    #     name=None,
    #     center=[0, 0, 0],
    #     theta=m1_theta,
    #     limPhysX=[-25.0, 25.0],
    #     limPhysY=[-175.0, 175.0],
    #     p=m1_p,
    #     q=m1_q_t,
    #     )
    
    # beamLine.mirror_conical_xrt = roes.ConicalMirror(
    #     bl=beamLine,
    #     name=None,
    #     center=[0, 0, 0],
    #     theta=m1_theta,
    #     limPhysX=[-25.0, 25.0],
    #     limPhysY=[-175.0, 175.0],
    #     )
        
    beamLine.mirror = beamLine.mirror_xmf
    # beamLine.mirror = beamLine.mirror_ellipsoid_xmf
    # beamLine.mirror = beamLine.mirror_conical_xrt

    beamLine.screen_t = rscreens.Screen(
        bl=beamLine,
        name="SCR",
        center=[0, scr_t_y, scr_t_z],
        z=[0, -np.sin(m1_theta), np.cos(m1_theta)])

    beamLine.screen_s = rscreens.Screen(
        bl=beamLine,
        name="SCR",
        center=[0, scr_s_y, scr_s_z],
        z=[0, -np.sin(m1_theta), np.cos(m1_theta)])
    
    return beamLine


def run_process(beamLine):
    geometricSource01beamGlobal01 = beamLine.geometricSource.shine()

    M1Param01beamGlobal01, M1Param01beamLocal01 = beamLine.mirror.reflect(
        beam=geometricSource01beamGlobal01)

    screen_t_beamLocal01 = beamLine.screen_t.expose(
        beam=M1Param01beamGlobal01)

    screen_s_beamLocal01 = beamLine.screen_s.expose(
        beam=M1Param01beamGlobal01)

    outDict = {
        'geometricSource01beamGlobal01': geometricSource01beamGlobal01,
        'M1Param01beamGlobal01': M1Param01beamGlobal01,
        'M1Param01beamLocal01': M1Param01beamLocal01,
        'screen_t_beamLocal01': screen_t_beamLocal01,
        'screen_s_beamLocal01': screen_s_beamLocal01
    }
    beamLine.prepare_flow()
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    Source = xrtplot.XYCPlot(
        beam=r"geometricSource01beamGlobal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f",
            limits=[-1, 1],
            unit="pm",
            factor=1e9),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            fwhmFormatStr=r"%.3f",
            limits=[source_z*1e9-1, source_z*1e9+1],
            offset=source_z*1e9,
            unit="pm",
            factor=1e9),
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

    tan_focus = xrtplot.XYCPlot(
        beam=r"screen_t_beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f",
            # limits=[-200, 200],
            unit="pm",
            factor=1e9),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            fwhmFormatStr=r"%.3f",
            # limits=[-200, 200],
            unit="pm",
            factor=1e9),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        aspect=r"auto",
        title=r"Tangential Focus")
    plots.append(tan_focus)

    sag_focus = xrtplot.XYCPlot(
        beam=r"screen_s_beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr=r"%.3f",
            # limits=[-200, 200],
            unit="pm",
            factor=1e9),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            fwhmFormatStr=r"%.3f",
            # limits=[-200, 200],
            unit="pm",
            factor=1e9),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        aspect=r"auto",
        title=r"Sagittal Focus")
    plots.append(sag_focus)

    return plots


def main():
    beamLine = build_beamline()
    # beamLine.glow()
    E0 = list(beamLine.geometricSource.energies)[0]
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=1,
        processes=4,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()
