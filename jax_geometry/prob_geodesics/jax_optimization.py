#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from jax_geometry.setup import *

from jax_geometry.manifolds import RiemannianManifold

#%% Gradient Descent Estimation of Geodesics

class JAXOptimization(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 score_fun:Callable,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lam:float=1.0,
                 lr_rate:float=1.0,
                 optimizer:Callable=None,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 )->None:
        
        self.M = M
        self.score_fun = score_fun
        
        self.lam = lam
        self.T = T
        self.max_iter = max_iter
        self.tol = tol
        
        if optimizer is None:
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr_rate)
        else:
            self.opt_init, self.opt_update, self.get_params = optimizer(lr_rate)
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
            
        self.z0 = None
        self.G0 = None
        self.zT = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def score_norm2(self, 
                    zt:Array,
                    )->Array:
        
        def score_norm2_path(score_norm2:Array,
                             z:Array,
                             )->Tuple[Array]:
            
            score = self.score_fun(z)

            score_norm2 += jnp.sum(score**2)
            
            return (score_norm2,)*2
        
        score_norm2, _ = lax.scan(score_norm2_path,
                                  init=self.score_norm20,
                                  xs=zt,
                                  )
        
        return score_norm2
    
    def energy(self, 
               zt:Array,
               )->Array:
        
        def energy_path(energy:Array,
                        y:Tuple,
                        )->Tuple[Array]:

            z, dz = y
            
            G = self.M.G(z)

            energy += jnp.einsum('i,ij,j->', dz, G, dz)
            
            return (energy,)*2

        term1 = zt[0]-self.z0
        energy_init = jnp.einsum('i,ij,j->', term1, self.G0, term1)
        zt = jnp.vstack((zt, self.zT))
        
        energy, _ = lax.scan(energy_path,
                             init=energy_init,
                             xs=(zt[:-1], zt[1:]-zt[:-1]),
                             )
        
        return energy
    
    def reg_energy(self, 
                   zt:Array,
                   *args,
                   )->Array:
        
        def reg_energy_path(reg_energy:Array,
                            y:Tuple,
                            )->Tuple[Array]:

            z, dz = y
            
            G = self.M.G(z)
            score = self.score_fun(z)

            reg_energy += jnp.einsum('i,ij,j->', dz, G, dz) + self.lam_norm*jnp.sum(score**2)
            
            return (reg_energy,)*2

        term1 = zt[0]-self.z0
        reg_energy_init = jnp.einsum('i,ij,j->', term1, self.G0, term1) + self.score_norm20
        zt = jnp.vstack((zt, self.zT))
        
        reg_energy, _ = lax.scan(reg_energy_path,
                                 init=reg_energy_init,
                                 xs=(zt[:-1], zt[1:]-zt[:-1]),
                                 )
        
        return reg_energy
    
    def Dregenergy(self,
                   zt:Array,
                   *args,
                   )->Array:
        
        return lax.stop_gradient(grad(self.reg_energy)(zt,*args))
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, Array, int],
                 )->Array:
        
        zt, grad, opt_state, idx = carry

        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple[Array, Array, Array, int],
                   )->Array:
        
        zt, grad, opt_state, idx = carry
        
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)

        grad = self.Dregenergy(zt)
        
        return (zt, grad, opt_state, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array, Array],
                 idx:int,
                 )->Array:
        
        zt, opt_state = carry
        
        grad = self.Dregenergy(zt)
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)
        
        return ((zt, opt_state),)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        self.z0 = z0
        self.zT = zT
        self.dim = len(z0)
        
        self.G0 = lax.stop_gradient(self.M.G(z0))
        self.score_norm20 = lax.stop_gradient(jnp.sum(self.score_fun(z0)**2))
        
        zt = self.init_fun(z0,zT,self.T)
        
        energy_init = self.energy(zt)
        score_norm2_init = self.score_norm2(zt)
        self.lam_norm = self.lam*energy_init/score_norm2_init
        
        opt_state = self.opt_init(zt)
        
        if step == "while":
            grad = self.Dregenergy(zt)
        
            zt, grad, _, idx = lax.while_loop(self.cond_fun, 
                                              self.while_step,
                                              init_val=(zt, grad, opt_state, 0)
                                              )
            reg_energy = self.reg_energy(zt)
            grad_norm = jnp.linalg.norm(grad)
        
            zt = jnp.vstack((z0, zt, zT))
        elif step == "for":
            _, val = lax.scan(self.for_step,
                              init=(zt, opt_state),
                              xs = jnp.ones(self.max_iter),
                              )
            
            zt = val[0]
            
            reg_energy = vmap(self.reg_energy)(zt)
            grad = vmap(self.Dregenergy)(zt)
            grad_norm = jnp.linalg.norm(grad, axis=1)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return zt, reg_energy, grad_norm, idx
    
#%% JAX Optimization with Euclidean Metric

class JAXEuclideanOptimization(ABC):
    def __init__(self,
                 score_fun:Callable,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lam:float=1.0,
                 lr_rate:float=1.0,
                 optimizer:Callable=None,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 )->None:
        
        self.score_fun = score_fun
        
        self.lam = lam
        self.T = T
        self.max_iter = max_iter
        self.tol = tol
        
        if optimizer is None:
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr_rate)
        else:
            self.opt_init, self.opt_update, self.get_params = optimizer(lr_rate)
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
            
        self.z0 = None
        self.G0 = None
        self.zT = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def score_norm2(self, 
                    zt:Array,
                    )->Array:
        
        def score_norm2_path(score_norm2:Array,
                             z:Array,
                             )->Tuple[Array]:
            
            score = self.score_fun(z)

            score_norm2 += jnp.sum(score**2)
            
            return (score_norm2,)*2
        
        score_norm2, _ = lax.scan(score_norm2_path,
                                  init=self.score_norm20,
                                  xs=zt,
                                  )
        
        return score_norm2
    
    def energy(self, 
               zt:Array,
               )->Array:

        zt = jnp.vstack((self.z0, zt, self.zT))
        ut = zt[1:]-zt[:-1]
        
        return jnp.sum(jnp.einsum('ti,ti->t', ut, ut))
    
    def reg_energy(self, 
                   zt:Array,
                   *args,
                   )->Array:

        score_norm2 = self.score_norm2(zt)
        energy = self.energy(zt)
        
        return energy+self.lam_norm*score_norm2
    
    def Dregenergy(self,
                   zt:Array,
                   *args,
                   )->Array:
        
        return lax.stop_gradient(grad(self.reg_energy)(zt,*args))
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, Array, int],
                 )->Array:
        
        zt, grad, opt_state, idx = carry

        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple[Array, Array, Array, int],
                   )->Array:
        
        zt, grad, opt_state, idx = carry
        
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)

        grad = self.Dregenergy(zt)
        
        return (zt, grad, opt_state, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array, Array],
                 idx:int,
                 )->Array:
        
        zt, opt_state = carry
        
        grad = self.Dregenergy(zt)
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)
        
        return ((zt, opt_state),)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        self.z0 = z0
        self.zT = zT
        self.dim = len(z0)

        self.score_norm20 = lax.stop_gradient(jnp.sum(self.score_fun(z0)**2))
        
        zt = self.init_fun(z0,zT,self.T)
        
        energy_init = self.energy(zt)
        score_norm2_init = self.score_norm2(zt)
        self.lam_norm = self.lam*energy_init/score_norm2_init
        
        opt_state = self.opt_init(zt)
        
        if step == "while":
            grad = self.Dregenergy(zt)
        
            zt, grad, _, idx = lax.while_loop(self.cond_fun, 
                                              self.while_step,
                                              init_val=(zt, grad, opt_state, 0)
                                              )
            reg_energy = self.reg_energy(zt)
            grad_norm = jnp.linalg.norm(grad)
        
            zt = jnp.vstack((z0, zt, zT))
        elif step == "for":
            _, val = lax.scan(self.for_step,
                              init=(zt, opt_state),
                              xs = jnp.ones(self.max_iter),
                              )
            
            zt = val[0]
            
            reg_energy = vmap(self.reg_energy)(zt)
            grad = vmap(self.Dregenergy)(zt)
            grad_norm = jnp.linalg.norm(grad, axis=1)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return zt, reg_energy, grad_norm, idx