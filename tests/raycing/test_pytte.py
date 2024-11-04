# -*- coding: utf-8 -*-
"""
Comparison tests for pyTTE backends
-------------------

1. Original pyTTE
Requires xraylib and multiprocess packages (both can be pip-installed), xraylib
provides the material properties. Solution of Takagi-Taupin equations relies on 
scipy.integrate.ODE, zvode-bdf algorithm. 
Fails on thick crystals (>1mm in case 2), requires increasing 'nsteps' in the
integrator up to 2-3 million in order to converge.
Calculation of transmitted intensity in Bragg geometry is not supported.

2. xrt pyTTE_x CPU
Pure python custom implementation of Dormand-Prince 4/5 adaptive algorithm.
Material properties backend can be either 'xrt' (default) or 'xraylib',
selectable via the TTcrystalX mat_backend init parameter. 
In our implementation the nominal radius of curvature used for strain
calculation is defined at the crystal's top surface, while in the original
pyTTE, it is centered within the crystal. This difference causes a slight
angular shift in the position of the reflectivity peak compared to the original
pyTTE implementation. Surface-centered radius of curvature is the standard
for the raycing backend. In this test the position of the radius of curvature
can be defined in the TTcrystalX class init by setting the 'strain_shift'
argument to either 'xrt' or 'pytte'.

3. xrt pyTTE_x OpenCL
Similar to pyTTE_x CPU, ported to execute on GPU with OpenCL. Calculation of
transmitted intensity in Bragg geometry is not supported.


"""
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "26 Oct 2024"

import numpy as np
import os, sys;
sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.materials_crystals as rmc
import xrt.backends.raycing as raycing  # analysis:ignore
from xrt.backends.raycing.pyTTE_x import TTcrystal as TTcrystalX
from xrt.backends.raycing.pyTTE_x import TTscan as TTscanX
from xrt.backends.raycing.pyTTE_x import Quantity as QuantityX
from xrt.backends.raycing.pyTTE_x.pyTTE_rkpy import TakagiTaupin as TakagiTaupinX
from pyTTE import TTcrystal, TTscan, Quantity, TakagiTaupin
from matplotlib import pyplot as plt

try:
    import xraylib  # analysis:ignore
    import multiprocess  # analysis:ignore
    isXrayLib = True
except ImportError:
    isXrayLib = False
    print("xraylib and multiprocess must be installed to run pyTTE")

try:
    import pyopencl as cl  # analysis:ignore
    import xrt.backends.raycing.myopencl as mcl
    targetOpenCL = 'auto'
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    xrt_cl = mcl.XRT_CL(r'materials.cl', precisionOpenCL='float32',
                        targetOpenCL=targetOpenCL)
    isOpenCL = hasattr(xrt_cl, 'cl_precisionF')
except ImportError:
    isOpenCL = False
    print("pyopencl and supported GPU/CPI driver are required for OpenCL")


case1 = {
    "name": "Bragg Flat",
    "crystal": "Si", # select from ('Si', 'Ge', 'Diamond', 'InSb', 'AlphaQuartz', 'Sapphire')
    "geometry": "Bragg reflected",  # Combination of Bragg/Laue reflected/transmitted
    "hkl": [1, 1, 1],
    "thickness": 1.,  # mm
    "asymmetry": 0.,  # degrees
    "in_plane_rotation": 0.,  # degrees
    "bending_Rm": np.inf,  # meters
    "bending_Rs": np.inf,  # meters
    "energy": 9000,  # eV
    "theta_array": np.linspace(-50, 150, 800),  # microrad
    "polarization": "sigma",  # or "pi"
    "precision": "float32"  # Only for opencl backend. "float64" for Laue
    }

case2 = {
    "name": "Bragg 1D bent",
    "crystal": "Si", # select from ('Si', 'Ge', 'Diamond', 'InSb', 'AlphaQuartz', 'Sapphire')
    "geometry": "Bragg reflected",  # Combination of Bragg/Laue reflected/transmitted
    "hkl": [1, 1, 1],
    "thickness": 0.75,  # mm
    "asymmetry": 5.,  # degrees
    "in_plane_rotation": 0.,  # degrees
    "bending_Rm": 10.,  # meters
    "bending_Rs": np.inf,  # meters
    "energy": 9000.,  # eV
    "theta_array": np.linspace(-50, 150, 800),  # microrad
    "polarization": "sigma",  # or "pi"
    "precision": "float32"  # Only for opencl backend. "float64" for Laue
    }

case3 = {
    "name": "Bragg 2D bent",
    "crystal": "Ge", # select from ('Si', 'Ge', 'Diamond', 'InSb', 'AlphaQuartz', 'Sapphire')
    "geometry": "Bragg reflected",  # Combination of Bragg/Laue reflected/transmitted
    "hkl": [3, 3, 3],
    "thickness": 0.5,  # mm
    "asymmetry": -2.,  # degrees
    "in_plane_rotation": 0.,  # degrees
    "bending_Rm": 10.,  # meters
    "bending_Rs": 10,  # meters
    "energy": 19000,  # eV
    "theta_array": np.linspace(-150, 50, 800),  # microrad
    "polarization": "sigma",  # or "pi"
    "precision": "float32"  # Only for opencl backend. "float64" for Laue
    }

case4 = {
    "name": "Laue 1D bent",
    "crystal": "Si", # select from ('Si', 'Ge', 'Diamond', 'InSb', 'AlphaQuartz', 'Sapphire')
    "geometry": "Laue reflected",  # Combination of Bragg/Laue reflected/transmitted
    "hkl": [1, 1, 1],
    "thickness": 0.5,  # mm
    "asymmetry": -12.,  # degrees
    "in_plane_rotation": 60.,  # degrees
    "bending_Rm": 10.,  # meters
    "bending_Rs": np.inf,  # meters
    "energy": 29000,  # eV
    "theta_array": np.linspace(-100, 100, 1500),  # microrad
    "polarization": "sigma",  # or "pi"
    "precision": "float64"  # Only for opencl backend. "float64" for Laue
    }

case5 = {
    "name": "Laue 2D bent",
    "crystal": 'Si', # select from ('Si', 'Ge', 'Diamond', 'InSb', 'AlphaQuartz', 'Sapphire')
    "geometry": "Laue reflected",  # Combination of Bragg/Laue reflected/transmitted
    "hkl": [1, 1, 1],
    "thickness": 2.,  # mm
    "asymmetry": 22.,  # degrees
    "in_plane_rotation": 60.,  # degrees
    "bending_Rm": 20.,  # meters
    "bending_Rs": 20,  # meters
    "energy": 60000,  # eV
    "theta_array": np.linspace(-100, 100, 1500),  # microrad
    "polarization": "sigma",  # or "pi"
    "precision": "float64"  # Only for opencl backend. "float64" for Laue
    }


def run_pytte_xrt_opencl(name, crystal, geometry, hkl, thickness, asymmetry,
                         in_plane_rotation, bending_Rm, bending_Rs, energy,
                         theta_array, polarization, precision):
    alpha = np.radians(asymmetry)
    crInstance = getattr(rmc, crystal)(hkl=hkl,
                                       geom=geometry,
                                       t=thickness,
                                       useTT=True)

    thetaCenter = np.arcsin(rm.ch / (2*crInstance.d*energy))
    E_in = np.ones_like(theta_array) * energy
    theta = theta_array*1e-6 + thetaCenter

    s0 = (np.zeros_like(theta), np.cos(theta+alpha), -np.sin(theta+alpha))
    sh = (np.zeros_like(theta), np.cos(theta-alpha), np.sin(theta-alpha))
    if geometry.startswith('Bragg'):
        n = (0, 0, 1)  # outward surface normal
    else:
        n = (0, -1, 0)  # outward surface normal
    hn = (0, np.sin(alpha), np.cos(alpha))  # outward Bragg normal
    gamma0 = sum(i*j for i, j in zip(n, s0))
    gammah = sum(i*j for i, j in zip(n, sh))
    hns0 = sum(i*j for i, j in zip(hn, s0))

    matCL = mcl.XRT_CL(r'materials.cl',
                       precisionOpenCL=precision,
                       targetOpenCL=targetOpenCL)

    ampS, ampP = crInstance.get_amplitude_pytte(
            E_in, gamma0, gammah, hns0, ucl=matCL, alphaAsym=alpha,
            inPlaneRotation=np.radians(in_plane_rotation),
            Ry=bending_Rm*1e3, Rx=bending_Rs*1e3)
    return np.abs(ampS)**2 if polarization == "sigma" else np.abs(ampP)**2


def run_pytte_xrt_cpu_rk45cpu(name, crystal, geometry, hkl, thickness,
                              asymmetry, in_plane_rotation, bending_Rm,
                              bending_Rs, energy, theta_array, polarization,
                              precision):

    alpha = np.radians(asymmetry)
    geotag = 0 if geometry.startswith('B') else np.pi*0.5
    crInstance = getattr(rmc, crystal)(hkl=hkl,
                                       geom=geometry,
                                       t=thickness)

    refl = geometry.endswith("lected")

    ttx = TTcrystalX(crystal=crystal,
                     hkl=hkl,
                     thickness=QuantityX(thickness, 'mm'),
                     debye_waller=1,
                     xrt_crystal=crInstance,
                     Rx=QuantityX(bending_Rm, 'm'),
                     Ry=QuantityX(bending_Rs, 'm'),
                     asymmetry=QuantityX(alpha+geotag, 'rad'),
                     in_plane_rotation=QuantityX(in_plane_rotation, 'deg'),
#                     mat_backend='xraylib',  # Default is 'xrt'
#                     strain_shift='pytte',  # Default is 'xrt'
                     )

    tts = TTscanX(constant=QuantityX(energy, 'eV'),
                  scan=QuantityX(theta_array, 'urad'),
                  polarization=polarization)

    scan_tt_s = TakagiTaupinX(ttx, tts)
    scan_vector, R, T, curSD = scan_tt_s.run()
    return R if refl else T


def run_pytte_original(name, crystal, geometry, hkl, thickness, asymmetry,
                       in_plane_rotation, bending_Rm, bending_Rs, energy,
                       theta_array, polarization, precision):
    alpha = np.radians(asymmetry)
    geotag = 0 if geometry.startswith('B') else np.pi*0.5
    refl = geometry.endswith("lected")

    ttx = TTcrystal(crystal=crystal,
                    hkl=hkl,
                    thickness=Quantity(thickness, 'mm'),
                    debye_waller=1,
                    Rx=Quantity(bending_Rm, 'm'),
                    Ry=Quantity(bending_Rs, 'm'),
                    asymmetry=Quantity(alpha+geotag, 'rad'),
                    in_plane_rotation=Quantity(in_plane_rotation, 'deg')
                    )

    tts = TTscan(constant=Quantity(energy, 'eV'),
                 scan=Quantity(theta_array, 'urad'),
                 polarization=polarization)

    scan_tt_s = TakagiTaupin(ttx, tts)
    scan_vector, R, T = scan_tt_s.run()

    return R if refl else T


if __name__ == '__main__':

    for icp, calcParams in enumerate(
            [case1, case2, case3, case4, case5]):
        plt.figure(calcParams["name"])
        for ifunc, func in enumerate([
                     run_pytte_original if isXrayLib else None,
                     run_pytte_xrt_cpu_rk45cpu,
                     run_pytte_xrt_opencl if isOpenCL else None
                     ]):
            if func:
                data = func(**calcParams)
                plt.plot(calcParams['theta_array'], data,
                         label=str(func.__name__).split("_")[-1],
                         linewidth=3-ifunc)
        plt.xlabel(r'$\theta-\theta_B$ ($\mu$rad)')
        plt.ylabel('|Amplitude|²')
        plt.title(calcParams["name"])    
        plt.legend()
        plt.savefig(f'{icp+1:02d} - {calcParams["name"]}.png')
        plt.show(block=False)
    plt.show()
