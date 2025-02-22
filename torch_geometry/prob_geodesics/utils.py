#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 21:17:01 2025

@author: fmry
"""

#%% Modules

import torch
from torch import Tensor

#%% Geodesic Optimization Module

class GeoCurve(torch.nn.Module):
    def __init__(self, 
                 z0:Tensor,
                 zt:Tensor,
                 zT:Tensor,
                 )->None:
        super(GeoCurve, self).__init__()
        
        self.z0 = z0
        self.zT = zT
        self.zt = torch.nn.Parameter(zt, requires_grad=True)
        
        return
    
    def ut(self,
           zt:Tensor,
           )->Tensor:
        
        zt = torch.vstack((self.z0, zt, self.zT))
        ut = zt[1:]-zt[:-1]
        
        return ut
    
    def forward(self, 
                )->Tensor:
        
        return self.zt