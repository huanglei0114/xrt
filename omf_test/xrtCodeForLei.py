# -*- coding: utf-8 -*-
# xrt scripts

# height error for parametric surfaces is not working yet ...

import sys
sys.path.append('/Users/dvorak/joePy/general/')
sys.path.append('/home/jdvorak/joePy')
# standard modules
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy.optimize import fsolve
# xrt code
import xrt.backends.raycing as raycing
import xrt.backends.raycing.oes as roe
from xrt.backends.raycing.oes_base import OE
import xrt.backends.raycing.materials as rm
# joe python code
# import zpEqns as zpEqns
# import gGrEqns as gGrEqns # joe's grating functions
# import diaboloid_borrowed_code as dbc

hc_eVmm = 12398.419297617678e-7

class EllipsoidMirrorJoe(OE):
    """Implements an ellipsoidal mirror. The ellipsoid is defined optically
    by p, q, and theta, where theta in the incident grazing angle in 
    radians. This is an ellipsoid of revolution."""

    def __init__(self, *args, **kwargs):
        # it looks to me like args and kwargs are passed to the base OE class,
        # after removing and assigning p,q, and theta to this class
        """
        *p*: source distance
        *r*: image distance
        *theta*: incident grazing angle in radians
        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        if self.p == None:
            raise ValueError("No value given for p")
        if self.q == None:
            raise ValueError("No value given for q")
        if self.t == None:
            raise ValueError("No value given for theta")

    def __pop_kwargs(self, **kwargs):
        self.p = kwargs.pop('p', None)
        self.q = kwargs.pop('q', None)
        self.t = kwargs.pop('theta', None)
        return kwargs

    def local_z(self, x, y):
        h = (self.p - self.q)*np.cos(self.t)
        A = h*h + 4*self.p*self.q
        B = 2*(self.p+self.q)*np.sin(self.t)*(h*y-2*self.p*self.q)
        C = (self.p+self.q)*(self.p+self.q)*(x*x+np.sin(self.t)*np.sin(self.t)*y*y)
        z = 0.5*(1/A)*(-B-np.sqrt(B*B-4*A*C))
        return z

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        K = (self.p+self.q)
        h = (self.p - self.q)*np.cos(self.t)
        z = self.local_z(x,y)
        dfdx = 2*K*K*x
        dfdy = 2*K*np.sin(self.t)*h*z + 2*K*K*np.sin(self.t)*np.sin(self.t)*y       
        dfdz = 2*(h*h+4*self.p*self.q)*z + 2*K*np.sin(self.t)*(h*y-2*self.p*self.q)
        norm = (dfdx**2 + dfdy**2 + dfdz**2)**0.5
        return [-dfdx/norm, -dfdy/norm, -dfdz/norm]

class ConeMirrorJoe(OE):
    """Implements an unbent right circular cone accorging to:
    Valeriy V. Yashchuk et al. Diaboloidal mirrors: algebraic solution and 
    surface shape approximations, J. Syn- chrotron Rad. (2021). 28, 1031–1040.
    The cone is defined optically by p, q, and theta, where theta in the 
    incident grazing angle in radians."""

    def __init__(self, *args, **kwargs):
        # it looks to me like args and kwargs are passed to the base OE class,
        # after removing and assigning p,q, and theta to this class
        """
        *p*: source distance
        *r*: image distance
        *theta*: incident grazing angle in radians
        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        if self.p == None:
            raise ValueError("No value given for p")
        if self.q == None:
            raise ValueError("No value given for q")
        if self.t == None:
            raise ValueError("No value given for theta")

    def __pop_kwargs(self, **kwargs):
        self.p = kwargs.pop('p', None)
        self.q = kwargs.pop('q', None)
        self.t = kwargs.pop('theta', None)
        return kwargs

    def local_z(self, x, y):
        p = self.p
        q = self.q
        t = self.t
        Ro = 2*p*q*np.cos(t)**2*np.sin(t)/(p+q)
        Ry = Ro*(1-((2*p*np.cos(t)**2-q)*y)/(2*p*q*np.cos(t)))
        z = Ry - np.sqrt(Ry*Ry - x*x)
        return z

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        p = self.p
        q = self.q
        t = self.t
        Ro = 2*p*q*np.cos(t)**2*np.sin(t)/(p+q)
        Ry = Ro*(1-((2*p*np.cos(t)**2-q)*y)/(2*p*q*np.cos(t)))
        dfdx = -x/np.sqrt(Ry*Ry-x*x)
        k = (2*p*q*np.cos(t)**2-q)/(2*p*q*np.cos(t))
        dfdy = Ro*k*(1-Ry/np.sqrt(Ry*Ry-x*x))   
        dfdz = 1
        norm = (dfdx**2 + dfdy**2 + dfdz**2)**0.5
        return [dfdx/norm, dfdy/norm, dfdz/norm]

class TanEllipticalCylinderJoe(OE):
    """Implements a tangential elliptical cylinder. The ellipse is defined 
    optically by p, q, and theta, where theta in the incident grazing angle in 
    radians."""
    
    # derived from EllipsoidMirrorJoe by setting x=0 in local_z and local_n

    def __init__(self, *args, **kwargs):
        # it looks to me like args and kwargs are passed to the base OE class,
        # after removing and assigning p,q, and theta to this class
        """
        *p*: source distance
        *q*: image distance
        *theta*: incident grazing angle in radians
        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        if self.p == None:
            raise ValueError("No value given for p")
        if self.q == None:
            raise ValueError("No value given for q")
        if self.t == None:
            raise ValueError("No value given for theta")

    def __pop_kwargs(self, **kwargs):
        self.p = kwargs.pop('p', None)
        self.q = kwargs.pop('q', None)
        self.t = kwargs.pop('theta', None)
        return kwargs

    def local_z(self, x, y):
        h = (self.p - self.q)*np.cos(self.t)
        A = h*h + 4*self.p*self.q
        B = 2*(self.p+self.q)*np.sin(self.t)*(h*y-2*self.p*self.q)
        C = (self.p+self.q)*(self.p+self.q)*(np.sin(self.t)*np.sin(self.t)*y*y)
        z = 0.5*(1/A)*(-B-np.sqrt(B*B-4*A*C))
        return z

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        K = (self.p+self.q)
        h = (self.p - self.q)*np.cos(self.t)
        z = self.local_z(x,y)
        dfdx = 0
        dfdy = 2*K*np.sin(self.t)*h*z + 2*K*K*np.sin(self.t)*np.sin(self.t)*y       
        dfdz = 2*(h*h+4*self.p*self.q)*z + 2*K*np.sin(self.t)*(h*y-2*self.p*self.q)
        norm = (dfdx**2 + dfdy**2 + dfdz**2)**0.5
        return [-dfdx/norm, -dfdy/norm, -dfdz/norm]

def local_n_numerical(self, x, y):
    """Determines the normal vector of OE at (x, y) position numerically."""
    # calculate normal numerically from surface function
    #    return z
    dx = 1e-5
    dy = 1e-5
    #   
    vxArr = np.zeros((3,x.size))
    vxArr[0,:] = dx
    vxArr[1,:] = 0
    vxArr[2,:] = self.local_z((x + dx),y)-self.local_z(x,y)    
    #   
    vyArr = np.zeros((3,x.size))
    vyArr[0,:] = 0
    vyArr[1,:] = dy
    vyArr[2,:] = self.local_z(x,(y + dy))-self.local_z(x,y)   
    #
    nArr = np.zeros((3,x.size))
    for j in range(x.size):
        vn = np.cross(vxArr[:,j],vyArr[:,j])
        nArr[:,j] = vn/np.linalg.norm(vn)     
    return nArr

############################################################################

class SagCollDiaboloidMirrorJoe(OE):
    """Implements a sagittal collimating diaboloid mirror. The diaboloid
    is defined optically by p, q, and theta, where theta in the incident 
    grazing angle in radians. """

    def __init__(self, *args, **kwargs):
        # it looks to me like args and kwargs are passed to the base OE class,
        # after removing and assigning p,q, and theta to this class
        """
        *p*: source distance
        *r*: image distance
        *theta*: incident grazing angle in radians
        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        if self.p == None:
            raise ValueError("No value given for p")
        if self.q == None:
            raise ValueError("No value given for q")
        if self.t == None:
            raise ValueError("No value given for theta")

    def __pop_kwargs(self, **kwargs):
        self.p = kwargs.pop('p', None)
        self.q = kwargs.pop('q', None)
        self.t = kwargs.pop('theta', None)
        return kwargs

    def local_z(self, x, y):         
        sin = np.sin(self.t)
        cos = np.cos(self.t)
        h = (self.p - self.q)*cos
        K = self.p + self.q
        p = self.p
        q = self.q
        A = h*h + 4*p*q
        B = -4*p*q*K*sin+2*(p*p-q*q)*sin*cos*y+(p-q)*sin*x*x
        C = K*(q-cos*y)*x*x + K*K*sin**2*y*y - 0.25*x**4
        z = 0.5*(1/A)*(-B - np.sqrt(B*B - 4*A*C))
        return z

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        sin = np.sin(self.t)
        cos = np.cos(self.t)
        h = (self.p - self.q)*cos
        K = self.p + self.q
        p = self.p
        q = self.q
        z = self.local_z(x,y)
        dfdx = 2*(p-q)*sin*x*z+2*K*(q-cos*y)*x-x**3
        dfdy = 2*(p*p-q*q)*sin*cos*z-K*cos*x*x+2*K*K*sin**2*y
        dfdz = 2*(h*h+4*p*q)*z-4*p*q*K*sin+2*(p*p-q*q)*sin*cos*y+(p-q)*sin*x*x
        norm = (dfdx**2 + dfdy**2 + dfdz**2)**0.5
        return [-dfdx/norm, -dfdy/norm, -dfdz/norm]  
    
    def local_n_numerical(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        # calculate normal numerically from surface function
        # VFM.local_z(self, x, y):         
        #    return z
        dx = 1e-5
        dy = 1e-5
        #   
        vxArr = np.zeros((3,x.size))
        vxArr[0,:] = dx
        vxArr[1,:] = 0
        vxArr[2,:] = self.local_z((x + dx),y)-self.local_z(x,y)    
        #   
        vyArr = np.zeros((3,x.size))
        vyArr[0,:] = 0
        vyArr[1,:] = dy
        vyArr[2,:] = self.local_z(x,(y + dy))-self.local_z(x,y)   
        #
        nArr = np.zeros((3,x.size))
        for j in range(x.size):
            vn = np.cross(vxArr[:,j],vyArr[:,j])
            nArr[:,j] = vn/np.linalg.norm(vn)     
        return nArr

class TanCollDiaboloidMirrorJoe(OE):
    """Implements a tangential collimating diaboloid mirror. The diaboloid
    is defined optically by p, q, and theta, where theta in the incident 
    grazing angle in radians. """

    def __init__(self, *args, **kwargs):
        # it looks to me like args and kwargs are passed to the base OE class,
        # after removing and assigning to this class
        """
        *p*: source distance
        *r*: image distance
        *theta*: incident grazing angle in radians
        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        if self.p == None:
            raise ValueError("No value given for p")
        if self.q == None:
            raise ValueError("No value given for q")
        if self.t == None:
            raise ValueError("No value given for theta")

    def __pop_kwargs(self, **kwargs):
        self.p = kwargs.pop('p', None)
        self.q = kwargs.pop('q', None)
        self.t = kwargs.pop('theta', None)
        return kwargs

    def local_z(self, x, y): 
        # from Yashshuk, J. Synchrotron Rad. (2021). 28, 1031–1040
        # implements a point to line segment sag focusing diaboloid
        cos = np.cos(self.t)
        cos2 = np.cos(2*self.t)        
        sin = np.sin(self.t)
        sin2 = np.sin(2*self.t) 
        p = self.p
        q = self.q
        
        # quartic equation coefficients
        A = -cos**4
        B = 4*(p-q)*cos**2*sin+4*cos**3*sin*y
        C = 4*q*((p+q)*cos**2+4*p*sin**2)\
            +2*cos*(q-3*p+(p-3*q)*cos2)*y\
            -6*cos**2*sin**2*y**2
        D = -16*p*q*(p+q)*sin+4*(p+q)*(2*p-q)*sin2*y\
            +2*(3*p+q+(3*q+p)*cos2)*sin*y**2\
            +4*cos*sin**3*y**3
        E = 4*(p+q)**2*x**2+4*q*(p+q)*sin**2*y**2\
            -4*(p+q)*cos*sin**2*y**3-sin**4*y**4

        # Yashchuk solution
        b = B/A
        c = C/A
        d = D/A
        e = E/A
        k = (8*c-3*b*b)/8
        m = (b*b*b-4*b*c+8*d)/8
        del0 = c*c-3*b*d+12*e
        del1 = 2*c*c*c-9*b*c*d+27*b*b*e+27*d*d-72*c*e
        # note: eqn for Q in paper is wrong, correct expression is here
        Q = (0.5*(del1+np.emath.sqrt(del1**2-4*del0**3)))**(1/3)
        S = 0.5*np.sqrt(-(2/3)*k+(1/3)*(Q+del0/Q))
        z = np.real_if_close(-b/4-S+0.5*np.emath.sqrt(-4*S*S-2*k+m/S))

        return np.real(z)

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        cos = np.cos(self.t)
        cos2 = np.cos(2*self.t)        
        sin = np.sin(self.t)
        sin2 = np.sin(2*self.t) 
        p = self.p
        q = self.q
        z = self.local_z(x,y)
        
        # quartic equation coefficients
        A = -cos**4
        B = 4*(p-q)*cos**2*sin+4*cos**3*sin*y
        C = 4*q*((p+q)*cos**2+4*p*sin**2)\
            +2*cos*(q-3*p+(p-3*q)*cos2)*y\
            -6*cos**2*sin**2*y**2
        D = -16*p*q*(p+q)*sin+4*(p+q)*(2*p-q)*sin2*y\
            +2*(3*p+q+(3*q+p)*cos2)*sin*y**2\
            +4*cos*sin**3*y**3

        # calc surface normal
        dfdx = 8*(p+q)*(p+q)*x
        dfdy = 4*cos**3*sin*z*z*z \
               +(2*cos*(q-3*p+(p-3*q)*cos2)-12*cos**2*sin**2*y)*z*z \
               +(4*(p+q)*(2*p-q)*sin2+4*(3*p+q+(3*q+p)*cos2)*sin*y+12*cos*sin**3*y*y)*z \
               +8*q*(p+q)*sin**2*y-12*(p+q)*cos*sin**2*y*y-4*sin**4*y*y*y   
        dfdz = 4*A*z*z*z+3*B*z*z+2*C*z+D
        norm = (dfdx**2 + dfdy**2 + dfdz**2)**0.5
        n = np.real_if_close([-dfdx/norm, -dfdy/norm, -dfdz/norm])
        return np.real(n)

class SemiGeneralDiaboloidMirrorJoe_Take3(OE):
    """Implements an approximate semi-general diaboloid mirror, with a 
    stigmatic source point and separate tangential and sagittal focus points. 
    The diaboloid is defined optically by p, qt, qs and theta, where theta in 
    the incident grazing angle in radians.
    
    This version uses the analytic expression for the surface normal
    
    Parameter list:
        *p*: source distance
        *qt*: tangential image distance
        *qs*: sagittal image distance
        *theta*: incident grazing angle in radians
    """

    def __init__(self, *args, **kwargs):
        # it looks to me like args and kwargs are passed to the base OE class,
        # after removing and assigning p,q, and theta to this class
        """
        *p*: source distance
        *qt*: tangential image distance
        *qs*: sagittal image distance
        *theta*: incident grazing angle in radians
        *b*: parameter which relates to the correlation between mirror x,y 
             and the x position along the tangential focus line
        *c*: parameter related to the change in path length along the 
             tangential focus
        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        if self.p == None:
            raise ValueError("No value given for p")
        if self.qt == None:
            raise ValueError("No value given for qt")
        if self.qs == None:
            raise ValueError("No value given for qs")
        if self.t == None:
            raise ValueError("No value given for theta")
        self.a = (self.qs - self.qt)/self.qs
        self.b = (self.qs - self.qt)/self.qs**2
        self.c = 0.5*(1/(self.qt-self.qs))

    def __pop_kwargs(self, **kwargs):
        self.p = kwargs.pop('p', None)
        self.qt = kwargs.pop('qt', None)
        self.qs = kwargs.pop('qs', None)
        self.t = kwargs.pop('theta', None)
        return kwargs

    def local_z(self, x, y):         
        sin = np.sin(self.t)
        cos = np.cos(self.t)
        p = self.p
        qt = self.qt
        qs = self.qs
        a = self.a
        b = self.b
        c = self.c
        # 
        g = (a+b*y)    
        # A
        A = -4*(c**2*g**4*x**4 + 2*c*(p+qt)*g**2*x**2 + (p+qt)**2-(p-qt)**2*sin**2)
        # B
        B = 4*sin*(
                  + c**2*(p+qt)*g**4*x**4
                  + (2*c*(p+qt)**2*g**2 + (p-qt)*(g**2-2*g))*x**2
                  + 2*(p+qt)*(2*p*qt-(p-qt)*cos*y)
                  )  
        # C
        Tx8 = c**4*g**8*x**8
        Tx6 = 2*c**2*g**4*( 2*c*(p+qt)*g**2 - g**2 + 2*g - 2 )*x**6
        Tx4 = g**2*( 
                    + 4*c**2*g**2*( (p**2+3*p*qt+qt**2) - (p-qt)*cos*y - y**2 )
                    - 4*c*(p+qt)*(g**2 - 2*g + 2)
                    + g**2 - 4*g + 4
                    )*x**4
        Tx2 = -4*(p+qt)*(
                  + (2*c*(p-qt)*g**2 + g**2 - 2*g)*cos*y
                  + 2*c*g**2*(y**2-p*qt)
                  + (p*g*(g-2) + (p+qt))
                 )*x**2
        Tx0 = -4*sin**2*(p+qt)**2*y**2
        C = Tx8 + Tx6 + Tx4 + Tx2 + Tx0
        z = 0.5*(1/A)*(-B + np.sqrt(B*B - 4*A*C))
        return z

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        # calculate normal numerically from surface function
        sin = np.sin(self.t)
        cos = np.cos(self.t)
        p = self.p
        qt = self.qt
        qs = self.qs
        a = self.a
        b = self.b
        c = self.c
        # 
        g = (a+b*y)    
        # A
        A = -4*(c**2*g**4*x**4 + 2*c*(p+qt)*g**2*x**2 + (p+qt)**2-(p-qt)**2*sin**2)
        # B
        B = 4*sin*(
                  + c**2*(p+qt)*g**4*x**4
                  + (2*c*(p+qt)**2*g**2 + (p-qt)*(g**2-2*g))*x**2
                  + 2*(p+qt)*(2*p*qt-(p-qt)*cos*y)
                  )  
        # C
        Tx8 = c**4*g**8*x**8
        Tx6 = 2*c**2*g**4*( 2*c*(p+qt)*g**2 - g**2 + 2*g - 2 )*x**6
        Tx4 = g**2*( 
                    + 4*c**2*g**2*( (p**2+3*p*qt+qt**2) - (p-qt)*cos*y - y**2 )
                    - 4*c*(p+qt)*(g**2 - 2*g + 2)
                    + g**2 - 4*g + 4
                    )*x**4
        Tx2 = -4*(p+qt)*(
                  + (2*c*(p-qt)*g**2 + g**2 - 2*g)*cos*y
                  + 2*c*g**2*(y**2-p*qt)
                  + (p*g*(g-2) + (p+qt))
                 )*x**2
        Tx0 = -4*sin**2*(p+qt)**2*y**2
        C = Tx8 + Tx6 + Tx4 + Tx2 + Tx0
        z = 0.5*(1/A)*(-B + np.sqrt(B*B - 4*A*C))
        #
        dAdx = -16*c*(c*g**4*x**3 + (p+qt)*g**2*x)
        dBdx = 8*sin*(
                    +2*c**2*(p+qt)*g**4*x**3
                    + (2*c*(p+qt)**2*g**2 + (p-qt)*(g**2-2*g))*x
                    )
        dCdx = +8*c**4*g**8*x**7 \
                +12*c**2*g**4*(2*c*(p+qt)*g**2 - g**2 + 2*g - 2)*x**5 \
                +4*g**2*(
                    4*c**2*g**2*((p**2 + 3*p*qt + qt**2) - (p-qt)*cos*y - y**2)
                    -4*c*(p+qt)*(g**2 - 2*g + 2) + g**2 - 4*g + 4
                    )*x**3 \
                -8*(
                    (p+qt)*(2*c*(p-qt)*g**2 + g**2 - 2*g)*cos*y
                    +2*c*g**2*(p+qt)*(y**2 - p*qt)
                    +(p+qt)*(p*g*(g-2) + (p+qt))                    
                    )*x
        #
        dAdy = -16*b*c*g*(c*g**2*x**4 + (p+qt)*x**2)
        dBdy = +8*sin*(
                +2*b*c**2*(p+qt)*g**3*x**4 
                +b*(2*c*(p+qt)**2*g + (p-qt)*(g-1))*x**2
                -(p+qt)*(p-qt)*cos
                )
        dCdy = +8*b*c**4*g**7*x**8 \
                +4*b*c**2*g**3*(6*c*(p+qt)*g**2 - 3*g**2 + 5*g - 4)*x**6 \
                +4*g*(
                    +c**2*g**2*(4*b*(p**2 + 3*p*qt + qt**2) - (p-qt)*cos*(4*b*y + g) - 4*b*y**2 - 2*g*y)
                    -2*b*c*(p+qt)*(2*g**2 - 3*g + 2)
                    +b*(g**2 - 3*g + 2)
                    )*x**4 \
                -4*(p+qt)*(
                    (2*c*(p-qt)*(2*b*g*y + g**2) + (2*b*g*y + g**2) -2*(b*y + g))*cos
                    +4*c*g*(b*y**2 + g*y - b*p*qt)
                    +2*b*p*(g-1)
                    )*x**2 \
                -8*sin**2*(p+qt)**2*y
        dFdx = dAdx*z**2 + dBdx*z + dCdx
        dFdy = dAdy*z**2 + dBdy*z + dCdy
        dFdz = 2*A*z + B
        norm = (dFdx**2 + dFdy**2 + dFdz**2)**0.5
        return [-dFdx/norm, -dFdy/norm, -dFdz/norm]  
    










