#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from jax_geometry.setup import *

from jax_geometry.manifolds import RiemannianManifold
from jax_geometry.line_search import Backtracking

#%% Gradient Descent Estimation of Geodesics

class ProbGEORCE(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 score_fun:Callable,
                 init_fun:Callable=None,
                 lam:float=1.0,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {'rho': 0.5},
                 )->None:
        
        self.M = M
        self.score_fun = score_fun
        
        self.lam = lam
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem with Probability Flow"
    
    def initialize(self,
                   )->Tuple:
        
        zt = self.init_fun(self.z0,self.zT,self.T)
        
        val =  jnp.vstack((self.z0, zt, self.zT))
        ut = val[1:]-val[:-1]
        
        return zt, ut
    
    def score_norm2(self, 
                    zt:Array,
                    )->Array:
        
        score = self.score_fun(lax.stop_gradient(zt))
        score_norm2 = jnp.sum(score**2)
        
        return score_norm2
    
    def energy(self, 
               zt:Array,
               )->Array:
        
        dz0 = zt[0]-self.z0
        e1 = jnp.einsum('i,ij,j->', dz0, self.G0, dz0)
        
        zt = jnp.vstack((zt, self.zT))
        Gt = vmap(self.M.G)(lax.stop_gradient(zt[:-1]))
        dzt = zt[1:]-zt[:-1]
        
        return e1+jnp.sum(jnp.einsum('...i,...ij,...j->...', dzt, Gt, dzt))
    
    def reg_energy(self, 
                   zt:Array,
                   *args,
                   )->Array:
        
        energy = self.energy(zt)
        score_norm2 = self.score_norm2(zt)
        
        return lax.stop_gradient(energy + self.lam_norm*score_norm2)
    
    def Dregenergy(self,
                   zt:Array,
                   ut:Array,
                   Gt:Array,
                   gt:Array,
                   )->Array:
        
        return gt+2.*(jnp.einsum('tij,tj->ti', Gt[:-1], ut[:-1])-jnp.einsum('tij,tj->ti', Gt[1:], ut[1:]))

    def inner_product(self,
                      zt:Array,
                      ut:Array,
                      )->Array:
        
        Gt = vmap(self.M.G)(zt)
        score = self.score_fun(zt)
        score_norm2 = jnp.sum(score**2)
        
        return jnp.sum(jnp.einsum('...i,...ij,...j->...', ut, Gt, ut))+self.lam_norm*score_norm2, Gt

    def gt(self,
           zt:Array,
           ut:Array,
           )->Tuple[Array]:
        
        gt, Gt = lax.stop_gradient(grad(self.inner_product)(zt,ut[1:]))
        Gt = jnp.vstack((self.G0.reshape(-1,self.M.dim,self.M.dim),
                         Gt))
        
        return gt, Gt
    
    def update_scheme(self, 
                      gt:Array, 
                      gt_inv:Array,
                      )->Array:
        
        g_cumsum = jnp.cumsum(gt[::-1], axis=0)[::-1]
        ginv_sum = jnp.sum(gt_inv, axis=0)
        
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', gt_inv[:-1], g_cumsum), axis=0)+2.0*self.diff
        
        muT = -jnp.linalg.solve(ginv_sum, rhs)
        mut = jnp.vstack((muT+g_cumsum, muT))
        
        return mut
    
    def update_xt(self,
                  zt:Array,
                  alpha:Array,
                  ut_hat:Array,
                  ut:Array,
                  )->Array:
        
        return self.z0+jnp.cumsum(alpha*ut_hat[:-1]+(1.-alpha)*ut[:-1], axis=0)
    
    def cond_fun(self, 
                 carry:Tuple,
                 )->Array:
        
        zt, ut, Gt, gt, gt_inv, grad_val, idx = carry
        
        grad_norm = jnp.linalg.norm(grad_val)

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple,
                     )->Array:
        
        zt, ut, Gt, gt, gt_inv, grad_val, idx = carry
        
        mut = self.update_scheme(gt, gt_inv)
        
        ut_hat = -0.5*jnp.einsum('tij,tj->ti', gt_inv, mut)
        tau = self.line_search(zt, grad_val, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)
        
        gt, Gt = self.gt(zt,ut)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.vstack((self.Ginv0, jnp.linalg.inv(Gt[1:])))
        grad_val = self.Dregenergy(zt,ut, Gt, gt)
        
        return (zt, ut, Gt, gt, gt_inv, grad_norm, idx+1)
    
    def for_step(self,
                 carry:Tuple,
                 idx:int,
                 )->Array:
        
        zt, ut, Gt, gt, gt_inv, grad_val = carry
        
        mut = self.update_scheme(gt, gt_inv)

        ut_hat = -0.5*jnp.einsum('tij,tj->ti', gt_inv, mut)
        tau = self.line_search(zt, grad_val, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)
        
        gt, Gt = self.gt(zt,ut)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.vstack((self.Ginv0, jnp.linalg.inv(Gt[1:])))
        grad_val = self.Dregenergy(zt,ut, Gt, gt)

        return ((zt, ut, Gt, gt, gt_inv, grad_val),)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_xt,
                                        **self.line_search_params,
                                        )
        
        self.z0 = lax.stop_gradient(z0)
        self.zT = lax.stop_gradient(zT)
        self.diff = self.zT-self.z0
        self.dim = len(z0)
        
        self.G0 = lax.stop_gradient(self.M.G(z0))
        self.Ginv0 = jnp.linalg.inv(self.G0).reshape(1,self.dim,self.dim)
        
        zt, ut = self.initialize()

        energy_init = lax.stop_gradient(self.energy(zt))
        score_norm2_init = lax.stop_gradient(self.score_norm2(zt))
        self.lam_norm = self.lam*energy_init/score_norm2_init
        
        gt, Gt = self.gt(zt,ut)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.vstack((self.Ginv0, jnp.linalg.inv(Gt[1:])))
        grad_val = self.Dregenergy(zt,ut, Gt, gt)
        
        if step == "while":
            zt, ut, Gt, gt, gt_inv, grad_val, idx = lax.while_loop(self.cond_fun, 
                                                                   self.while_step, 
                                                                   init_val=(zt, 
                                                                             ut, 
                                                                             Gt, 
                                                                             gt, 
                                                                             gt_inv, 
                                                                             grad_val, 
                                                                             0),
                                                                   )
        elif step == "for":
            (zt, ut, Gt, gt, gt_inv, grad_val), _ = lax.scan(self.for_step,
                                                             init=(zt, ut, Gt, gt, gt_inv, grad_val),
                                                             xs=jnp.ones(self.max_iter),
                                                             )
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        reg_energy = self.reg_energy(zt)
        grad_norm = jnp.linalg.norm(grad_val)
        zt = jnp.vstack((z0, zt, zT))
            
        return zt, reg_energy, grad_norm, idx

#%% Probabilistic GEORCE for Euclidean Background Metric

class ProbEuclideanGEORCE(ABC):
    def __init__(self,
                 score_fun:Callable,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lam:float=1.0,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {'rho': 0.5},
                 )->None:

        self.score_fun = score_fun
        
        self.lam = lam
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem with Probability Flow"
    
    def initialize(self,
                   )->Tuple:
        
        zt = self.init_fun(self.z0,self.zT,self.T)
        
        val =  jnp.vstack((self.z0, zt, self.zT))
        ut = val[1:]-val[:-1]
        
        return zt, ut
    
    def score_norm2(self, 
                    zt:Array,
                    )->Array:
        
        score = self.score_fun(lax.stop_gradient(zt))
        score_norm2 = jnp.sum(score**2)
        
        return score_norm2
    
    def energy(self, 
               zt:Array,
               )->Array:

        zt = jnp.vstack((self.z0, lax.stop_gradient(zt), self.zT))
        ut = zt[1:]-zt[:-1]
        
        return jnp.sum(ut*ut)
    
    def reg_energy(self, 
                   zt:Array,
                   *args,
                   )->Array:

        score_norm2 = self.score_norm2(zt)
        energy = self.energy(zt)
        
        return lax.stop_gradient(energy+self.lam_norm*score_norm2)

    def Dregenergy(self,
                   ut:Array,
                   gt:Array,
                   )->Array:
        
        return gt+2.*(ut[:-1]-ut[1:])

    def inner_product(self,
                      zt:Array,
                      )->Array:

        score = self.score_fun(zt)
        score_norm2 = jnp.sum(score**2)
        
        return self.lam_norm*score_norm2
    
    def gt(self,
           zt:Array,
           )->Array:
        
        return lax.stop_gradient(grad(self.inner_product)(zt))
    
    def update_xt(self,
                  zt:Array,
                  alpha:Array,
                  ut_hat:Array,
                  ut:Array,
                  )->Array:
        
        return self.z0+jnp.cumsum(alpha*ut_hat[:-1]+(1.-alpha)*ut[:-1], axis=0)
    
    def update_ut(self,
                  gt:Array,
                  )->Array:
        
        g_cumsum = jnp.vstack((jnp.cumsum(gt[::-1], axis=0)[::-1], jnp.zeros(self.dim)))
        g_sum = jnp.sum(g_cumsum, axis=0)/self.T
        
        return self.diff/self.T+0.5*(g_sum-g_cumsum)

    def cond_fun(self, 
                 carry:Tuple,
                 )->Array:
        
        zt, ut, gt, grad_val, idx = carry
        
        grad_norm = jnp.linalg.norm(grad_val)

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple,
                     )->Array:
        
        zt, ut, gt, grad_val, idx = carry
        
        ut_hat = self.update_ut(gt)
        tau = self.line_search(zt, grad_val, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)
        
        gt = self.gt(zt)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_val = self.Dregenergy(ut, gt)
        
        return (zt, ut, gt, grad_val, idx+1)
    
    def for_step(self,
                 carry:Tuple,
                 idx:int,
                 )->Array:
        
        zt, ut, gt, grad_val = carry

        ut_hat = self.update_ut(gt)
        tau = self.line_search(zt, grad_val, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)
        
        gt = self.gt(zt)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_val = self.Dregenergy(ut, gt)

        return ((zt, ut, gt, grad_val),)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_xt,
                                        **self.line_search_params,
                                        )
        
        self.z0 = lax.stop_gradient(z0)
        self.zT = lax.stop_gradient(zT)
        self.diff = self.zT-self.z0
        self.dim = len(z0)
        
        zt, ut = self.initialize()

        energy_init = lax.stop_gradient(self.energy(zt))
        score_norm2_init = lax.stop_gradient(self.score_norm2(zt))
        self.lam_norm = self.lam*energy_init/score_norm2_init
        
        gt = self.gt(zt)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_val = self.Dregenergy(ut, gt)
        
        if step == "while":
            
            zt, ut, gt, grad_val, idx = lax.while_loop(self.cond_fun, 
                                                       self.while_step, 
                                                       init_val=(zt, 
                                                                 ut, 
                                                                 gt, 
                                                                 grad_val, 
                                                                 0),
                                                       )
        elif step == "for":
                
            (zt, ut, gt, grad_val), _ = lax.scan(self.for_step,
                                                 init=(zt, ut, gt, grad_val),
                                                 xs=jnp.ones(self.max_iter),
                                                 )
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        reg_energy = self.reg_energy(zt)
        grad_norm = jnp.linalg.norm(grad_val)
        zt = jnp.vstack((z0, zt, zT))
            
        return zt, reg_energy, grad_norm, idx
        