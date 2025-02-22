#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 01:05:15 2025

@author: fmry
"""

#%% Modules

import jax.numpy as jnp

from jax import Array
from abc import ABC

#%% Spherical Interpoalation

class SphericalInterpolation(ABC):
    def __init__(self,
                 T:int=100,
                 )->None:
        
        self.T = T
        
        return
    
    def __call__(self,
                 z0:Array,
                 zT:Array,
                 )->Array:
        
        shape = z0.shape
        
        z0 = z0.reshape(-1)
        zT = zT.reshape(-1)
        
        z0_norm = jnp.linalg.norm(z0)
        zT_norm = jnp.linalg.norm(zT)
        dot_product = jnp.dot(z0, zT)
        theta = jnp.arccos(dot_product/(z0_norm*zT_norm))
        
        sin_theta = jnp.sin(theta)
        
        t = jnp.linspace(0.0,1.0,self.T,endpoint=False,dtype=z0.dtype)[1:].reshape(-1,1)
        
        curve = ((z0*jnp.sin((1.-t)*theta) + zT*jnp.sin(t*theta))/sin_theta)
        
        curve = jnp.vstack((z0, curve, zT))
        
        return curve.reshape(-1, *shape)