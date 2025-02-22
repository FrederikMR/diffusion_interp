#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 01:05:15 2025

@author: fmry
"""

#%% Modules

import torch

from torch import Tensor

from abc import ABC

#%% Spherical Interpoalation

class SphericalInterpolation(ABC):
    def __init__(self,
                 T:int=100,
                 )->None:
        
        self.T = T
        
        return
    
    def __call__(self,
                 z0:Tensor,
                 zT:Tensor,
                 )->Tensor:
        
        shape = z0.shape
        
        z0 = z0.reshape(-1)
        zT = zT.reshape(-1)
        
        z0_norm = torch.linalg.norm(z0)
        zT_norm = torch.linalg.norm(zT)
        dot_product = torch.dot(z0, zT)
        theta = torch.arccos(dot_product/(z0_norm*zT_norm))
        
        sin_theta = torch.sin(theta)
        
        t = torch.linspace(0,1,self.T+1)[1:-1].reshape(-1,1)
        
        curve = ((z0*torch.sin((1.-t)*theta) + zT*torch.sin(t*theta))/sin_theta)
        
        curve = torch.vstack((z0, curve, zT))
        
        return curve.reshape(-1, *shape)