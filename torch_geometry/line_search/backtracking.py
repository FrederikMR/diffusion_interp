#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch

from torch import Tensor

from typing import Callable
from abc import ABC
    
#%% Backtracking Line Search

class Backtracking(ABC):
    def __init__(self,
                 obj_fun:Callable,
                 update_fun:Callable,
                 alpha:float=1.0,
                 rho:float=0.9,
                 c1:float=0.90,
                 max_iter:int=100,
                 )->None:
        #https://optimization.cbe.cornell.edu/index.php?title=Line_search_methods
        
        self.obj_fun = obj_fun
        self.update_fun = update_fun
        
        self.alpha = alpha
        self.rho = rho
        self.c1 = c1
        self.max_iter = max_iter
        
        self.x = None
        self.obj0 = None
        
        return
    
    @torch.no_grad()
    def armijo_condition(self, x_new:Tensor, obj:Tensor, alpha:Tensor, *args)->bool:
        
        val1 = self.obj0+self.c1*alpha*torch.dot(self.pk, self.grad0)
        
        return obj>val1
    
    @torch.no_grad()
    def cond_fun(self, 
                 alpha,
                 idx,
                 *args,
                 )->Tensor:

        x_new = self.update_fun(self.x, alpha, *args)
        obj = self.obj_fun(x_new, *args).item()
        bool_val = self.armijo_condition(x_new, obj, alpha, *args)
        
        return (bool_val) & (idx < self.max_iter)
    
    def update_alpha(self,
                     alpha:float,
                     idx:int,
                     )->Tensor:

        return self.rho*alpha, idx+1
    
    def __call__(self, 
                 x:Tensor,
                 grad_val:Tensor,
                 *args,
                 )->Tensor:
        
        self.x = x
        self.obj0 = self.obj_fun(x,*args).item()
        self.pk = -grad_val
        self.grad0 = grad_val
        
        alpha, idx = self.alpha, 0
        while self.cond_fun(alpha, idx, *args):
            alpha, idx = self.update_alpha(alpha, idx)

        return alpha