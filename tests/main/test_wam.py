from datetime import timedelta
from math import floor
from pathlib import Path

import yaml

from pyfv3 import DynamicalCoreConfig

# JK NOTE TODO: Just sticking things in here for now, will distribute them into their right
# places in the future.

def test_enable_wam() -> None:
    # Set up dycore config with enable_wam = True 
    # Set up dycore
    # Check that gravity is variable and grav_var and grav_var_h are used somehow?
    # maybe call to compute_geopotential?

    # Set up dycore config with enable_wam = False
    # Set up dycore
    # Check that gravity is a constant... somehow
    # maybe call to compute_geopotential?

    # TODO assert that something is different between wam_enabled = False
    # JK NOTE: I have to find out what to test.....
    
    assert false # TODO

############################ dyncore_state.py
def test_dycore_state() -> None:
    # Check that grav_var_h and grav_var both exist and are 3 dim?

    # JK NOTE: Don't know if this is even necessary. Check if other dycore state quantities
    # are tested in this way?
    assert false # TODO

############################ dyn_core.py
def test_average_gravity() -> None:
    # Check that average gravity values are reasonable
    # Use
    
    #def average_gravity(grav_var: FloatField, grav_var_h: FloatField):
    #"""
    #Args:
    #    grav_var (out): gravity field
    #    grav_var_h (in): gravity value at height
    #"""
    #with computation(FORWARD), interval(...):
    #    grav_var[0, 0, 0] = 0.5*(grav_var_h[0, 0, 0] + grav_var_h[0, 0, 1])
    assert false # TODO

def test_compute_geopotential() -> None:
    # Check that the change from constants.GRAV to grav_var_h are reasonable.
    
    #def compute_geopotential(zh: FloatField, gz: FloatField, grav_var_h: FloatField):
    #with computation(PARALLEL), interval(...):
    #    gz = zh * grav_var_h
    assert false # TODO

def test_p_grad_c_stencil() -> None:
    # Check that
    # 1. addition of average_gravity stencil and
    # 2. addition of grav_var_h parameter in self._compute_geopotential_stencil
    # still produces reasonable results
    
    assert false # TODO

############################ fv_dynamics.py

def test_init_gravity() -> None:
    # Check that init_gravity sets 3d grav_var to the constant GRAV for all vals
    assert false # TODO

def test_init_gravity_h() -> None:
    # Check that init_gravity sets 3d grav_var_h to constant GRAV for all vals
    assert false # TODO

def test_adjust_gravity() -> None:
    # Check that adjust_gravity sets grav_var and grav_var_h are set appropriately
    # with computation(FORWARD), interval(-1,None):
    #     newrad = RADIUS + (phis/GRAV)
    #     grav_var_h = GRAV*(RADIUS**2)/newrad**2
    # with computation(BACKWARD), interval(...):
    #    newrad = newrad - delz
    #    grav_var_h = GRAV*(RADIUS**2)/newrad**2
    #    grav_var = 0.5*(grav_var_h[0, 0, 1] + grav_var_h[0, 0, 0])

    assert false # TODO

# TODO JK NOTE to self --- checkout log_on_rank_0 for values that might be useful for test (possibly)
