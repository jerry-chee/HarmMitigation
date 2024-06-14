import copy
import random,itertools
import numpy as np
from scipy.linalg import solve

from abc import ABC, abstractmethod


def gradTester(f,x,eps = 0.001,delta=0.01):
    """Test gradient computations.
    """
    y,grd = f(x,True)
    
    if len(grd.shape)>1:
        n,d = grd.shape
    else:
        n = 1
        d = len(grd)

    dy = np.zeros((n,d))

    for j in range(d):
        e = np.zeros(d)
        e[j] = eps
        dyj = (f(x+e,grad=False) - y)/eps
        dy[:,j] = dyj   
    
    diff = dy-grd
    mse= np.multiply(diff,diff).sum()/n/d
    print("Grad ",type(f).__name__," MSE = ",mse, "<=delta", mse<=delta)
    # assert mse<=delta, "Expected grad error <= {delta}"


class Fun(ABC):
    """ Abstract function class. It defines an object that is callable and has a gradient/Jacobian. Inputs are always numpy arrays, and Jacobians are always nympy matrices (even if output has dimension 1). Outputs are either scalars or numpy arrays, depending on output size.
    """
    @abstractmethod
    def __call__(self, x, grad=False):
        """ If grad is false, the function only returns the function value f(x). If grad is True, it returns f(x), Jf(x), where J is the Jacobian. 
        """
        pass


class Compose(Fun):
    """ Composition. Given two functions at construction time, it produces a new function that is their composition. This is an example of how two Fun objects can be chained together via composition, to create a new function; the Jacobian will be the matrix product of their constituent Jacobians.
    """

    def __init__(self,fun_1,fun_2):
        """Parameters are two Fun objects to be composed.
        """
        self.fun_1 = fun_1
        self.fun_2 = fun_2

    def __call__(self,x,grad = False):
        """ When you call the function object, the value is the composition of the constituent function values, and the Jacobian is the matrix product of the Jacobians.
        """
        if grad:
            y_2,grad_2 = self.fun_2(x,grad)
            y_1,grad_1 = self.fun_1(y_2,grad)
            return y_1,grad_1 @ grad_2 
        else:
            y_2 = self.fun_2(x,grad)
            y_1 = self.fun_1(y_2,grad)
            return y_1
            

class ScoreGenerator(Fun):
    """ A class that generates score functions.
    """
    
    def __init__(self,V):
        """ Scores are parametrized by a numpy matrix V, containing item vectors.
        """
        self.V = V
        self.n,self.d=V.shape 
    
    def __call__(self,u,grad = False):
        """ Return the scores and, optionaly, their gradient, as a function of user profile u (a numpy array). 
        """
        s =  np.exp(np.dot(self.V, u))
        if grad:
            return s, self.V * s.reshape(len(s),1)
        else:
            return s
 

class P_vE_Generator(Fun):
    """A class that generates the selection probabilities  of items given different sets of recommendations.

    """
    def __init__(self,n,E_set,c=1):
        """ The selection probability is parametrized by the number of items n, the set of possible recommendations E_set, and the "premium" parameter c. The set E_set is a collection of subsets of item indices.
        """
        self.n = n
        self.E_set=sorted(E_set)
        self.E_set_to_index = dict(zip(self.E_set,range(len(self.E_set))))
        self.m = len(self.E_set)
        self.c=c

    def pos(self,vi,E):
        return vi*self.m+self.E_set_to_index[E]


    def __call__(self,s,grad = False):
        """Return the probabilities that item is selected (and optionally, its gradient) under this given recommendation, as a function of item scores s. """
        c = self.c
        E_set = self.E_set
        n = self.n
        m = self.m
        
        val = np.zeros(n*m)

        if grad:
            grd = np.zeros((n*m,n))
    
        s_Omega = sum(s)

        #print(n)
        #print(len(s))

        for vi in range(n):
            s_vi = s[vi]
           
            for E in E_set:
                s_E = sum( [s[vj]  for vj in E] )
        
                
                val[self.pos(vi,E)] = c*s_vi/(s_E+c)/(s_Omega)
        
                if vi in E:
                    val[self.pos(vi,E)] += s_vi/(s_E+c)     
            
                if grad:
                    for vj in range(n):
                        if vj in E:
                            grd[self.pos(vi,E)][vj] = -c*s_vi*(s_Omega+s_E+c)/(s_E+c)**2/(s_Omega)**2
                        if  vj not in E:
                            grd[self.pos(vi,E)][vj] = -c*s_vi/(s_E+c)/(s_Omega)**2
                        if vi == vj:
                            grd[self.pos(vi,E)][vj] += c/s_Omega/(s_E+c)

                        if vi in E:
                            if vj in E:
                                grd[self.pos(vi,E)][vj] += -s_vi/(s_E+c)**2
                            if vi==vj:
                                grd[self.pos(vi,E)][vj] +=  1/(s_E+c)

        if not grad:
            return val
        else:
            return val,grd


class P_Generator(Fun): 
    """ A generator producing selection probabilities.
    """

    def __init__(self,p_vE,s):
        """ Parameters is a conditional probability function and a score function, that are composed to produce the conditional probability function used in computing the probabilities.
        """
        self.n = p_vE.n
        self.m = p_vE.m
        self.d = s.d
        self.E_set=p_vE.E_set
        self.E_set_to_index = p_vE.E_set_to_index  
        self.pos = p_vE.pos
 
        self.p_vE=Compose(p_vE,s)

    def __call__(self,param,u,param_grad =False, u_grad=False):
        """ A function call. "Param" are the parameters of the distribution (here, the probabilities per set). "u" is the user profile. Grad computation w.r.t. either of these algorithms is optional.
        """
        n = self.n
        m = self.m
        d = self.d
        p_vE = self.p_vE

        vals = np.zeros(n)
    
        if param_grad:
            prm_grd = np.zeros((n,m))
    
        if u_grad:
            u_grd = np.zeros((n,d))
            p_vals,p_grads = p_vE(u,u_grad)
        else:
            p_vals = p_vE(u)
        
        for vi in range(n):
            start = self.E_set[0]
            end = self.E_set[-1]

            p_vi_vals = p_vals[self.pos(vi,start):self.pos(vi,end)+1]
            vals[vi] = np.dot(param,p_vi_vals)

            if param_grad:
                prm_grd[vi] = p_vi_vals
            
            if u_grad:
                p_vi_grads = p_grads[self.pos(vi,start):self.pos(vi,end)+1]
                u_grd[vi] = np.dot(param,p_vi_grads)
            
        if not (param_grad or u_grad):
            return vals
        else:
            res = [vals]
            if param_grad:
                res.append(prm_grd)
            if u_grad:
                res.append(u_grd)
            return tuple(res)    


    def fix_param(self,param):
        """Return a one-parameter function w.r.t. u, with param fixed.
        """
        return lambda u,grad: self.__call__(param,u,param_grad = False,u_grad=grad)

    def fix_u(self,u):
        """Return a one-parameter function w.r.t. p, with u fixed.
        """
        return lambda param,grad: self.__call__(param,u,param_grad = grad,u_grad=False)


class G_Generator(Fun):
    """Generator for G functions.
    """

    def __init__(self,n,HM,a_HM,a_NH,beta,u0,V):
        """Params are number of items, harm set, harm and non-harm attraction params, beta, and the inherent user profile u0, and the items V
        """
        self.n = n
        self.HM = HM
        self.a_HM = a_HM
        self.a_NH = a_NH
        self.beta = beta
        self.u0 =u0
        self.V = V

    def __call__(self,p,grad=False):
        """Compute G as a function of selection probabilities p.
        """
        n = self.n 
        HM = self.HM 
        a_HM = self.a_HM 
        a_NH = self.a_NH
        beta = self.beta 
        u0 = self.u0 
        V = self.V
        d = len(u0)
 
        NH = [vi for vi in  range(n) if vi not in HM]
        p_HM = sum( [p[vi] for vi in HM] ) 
        p_NH = sum( [p[vi] for vi in NH] ) 

        v_avg_HM =  sum( [p[vi]*V[vi,:] for vi in HM] )
        v_avg_NH = sum( [p[vi]*V[vi,:] for vi in NH] ) 

        num = beta*u0 + a_HM * v_avg_HM   + a_NH * v_avg_NH 
        denom = beta +  a_HM * p_HM +  a_NH * p_NH
        val = num/denom

        if grad:
            grd = np.zeros((d,n))

            for vi in range(n):
                v = V[vi,:]

                if vi in HM:
                    a_v = a_HM
                else:
                    a_v = a_NH

                d_num = a_v * denom * v  - num * a_v
                d_denom = denom**2

                grd[:,vi] = d_num/d_denom
        
        if grad:
            return val,grd
        else:
            return val
            

class F_Generator(Fun):
    """Generator of fixed point function F."""

    def __init__(self,G,p):
        """Inputs are a G function and a p (probabilities) function.
        """
        self.G = G
        self.p = p
        
    def fix_param(self,param):
        """Return a one-parameter function w.r.t. u, with param fixed.
        """
        return Compose(self.G,self.p.fix_param(param))

    def fix_u(self,u):
        """Return a one-parameter function w.r.t. p, with u fixed.
        """
        return Compose(self.G,self.p.fix_u(u))

    def __call__(self, param,u,param_grad =False, u_grad=False):
        """ Call F, as a function of param and u. Either gradients are optional
        """

        if not (param_grad or u_grad):
            return self.G(self.p(param,u))

        if param_grad and u_grad:
            p_val,prm_grd,u_grd = self.p(param,u,param_grad = param_grad,u_grad = u_grad)
            G_val,G_grd = self.G(p_val,grad=True)
            return G_val, G_grd @ prm_grd, G_grd @ u_grd

        if param_grad:
            F = self.fix_u(u)
            return F(param,grad = True)

        if u_grad:
            F = self.fix_param(param)
            return F(u,grad = True)

class U_MapGenerator(Fun):
    """Generator of function that computes the fixed-point  u = F(param,u)
    """
    
    def __init__(self,F,u0,eps = 0.00000001,max_iter=100):
        self.F = F
        self.u0 = u0
        self.eps = eps
        self.max_iter = max_iter

    def __call__(self,param,grad = False):
        """Given param, it computes the map param |--> u^*(param), where u^* is the fixed point satisfying:
            
              u^* = F(param,u^*)
        """
        err = np.Inf
        u = self.u0
        d = len(u)
        i = 0 
        F_par = self.F.fix_param(param)

        while(err > self.eps and i<self.max_iter):
            u_new = F_par(u,False)
            # err = np.linalg.norm(u-u_new)
            err = np.linalg.norm(u-u_new) / np.linalg.norm(u)
            u = u_new
            i += 1
        

        # assert i < self.max_iter, "Fixed point operation did not converge"
        if i >= self.max_iter:
            import warnings
            warnings.warn("Fixed point operation did not converge")            

        if not grad:
            return u
        else:
            val, par_grd, u_grd = self.F(param,u,param_grad = True, u_grad = True)
            A = u_grd - np.eye(d)  
            b = par_grd
            grd = - solve(A,b)
            return u, grd
  

class BoundedCardinalityModelLoss(Fun):
    """A class that generates the loss function.

    """
   
    def __init__(self,V,E_set,HM,a_HM,a_NH,beta,u0,u0_id,c=1,lam=0,eps = 0.00000001,max_iter=100,restart_umap = False):
        """Parameters are:
            - V: The item matrix V
            - E_set: The set of possible recommendations 
            - HM: The set of harmful items 
            - a_HM,a_NH,beta: the attraction model parameters
            - u0: The inherent profile 
            - u0_id: ID of inherent profile
            - c: the recommendation premium parameter
            - lam: the regularization parameter between the two objectives
            - eps: the tolerance of the fixed point convergence
            - max_iter: the maximum number of iterations of the fixed point convergence
            - restart_umap: restart umap after every call

        """
        self.V = V
        self.n = V.shape[0] 
        self.d = V.shape[1]
        self.c = c 
        self.HM = HM
        self.lam = lam
        self.eps = eps
        self.max_iter = max_iter
        self.u0_id = u0_id
        self.restart_umap = restart_umap
        
        self.s = ScoreGenerator(V)
        self.G = G_Generator(self.n,HM,a_HM,a_NH,beta,u0,V)
        self.p_vE = P_vE_Generator(self.n,E_set,c = self.c)
        self.p = P_Generator(self.p_vE,self.s)
        self.F = F_Generator(self.G,self.p)
        self.u_map = U_MapGenerator(self.F,u0,eps = self.eps, max_iter = self.max_iter) 
        
        self.E_set=sorted(E_set)
        self.E_set_to_index = dict(zip(E_set,range(len(E_set))))
        self.m = len(self.E_set)

    def pos(self,E):
        return self.E_set_to_index[E]

    def set_u_start(self,u_start):
        """ Redefine u_start. Can be useful during gradient descent
        """
        self.u_map = U_MapGenerator(self.F,u_start,eps = self.eps, max_iter = self.max_iter)

    
    def __call__(self,param,grad = False):
        """Return the loss (and optionally, its gradient) under this given recommendation, as a function of item scores s. """
        # print("called bounded cardinality loss")
        # Find fixed point u under input parameters and its gradient/sensitivity
        u_bar, u_bar_grad = self.u_map(param,True)

        # Store u_bar for future use, in case it is needed
        self.u_bar = u_bar
        self.u_bar_grad = u_bar_grad

        
        # Compute item scores under this user profile
        scores = self.s(u_bar)
    
        # Compute p_click and its gradient

        def g(s):
            return s/(s+self.c)
    
        def dg(s):
            return self.c/(s+self.c)**2

        set_scores = {}    

        if grad:
            set_scores_grd = {}
            hm_scores_grd_u = sum(  [scores[v]*self.V[v]  for v in self.HM])
            omega_scores_grd_u = sum(  [scores[v]*self.V[v]  for v in range(self.n)])

        for E in self.E_set:
            set_scores[E] = sum( [scores[v]  for v in E] )
            if grad:
                set_scores_grd[E] =   sum( [scores[v]*self.V[v]  for v in E] )

        p_clk_val = sum( [ param[self.pos(E)] * g(set_scores[E]) for E in self.E_set] )
            
        if grad:
            p_clk_grd_pi = np.zeros(self.m)
            p_clk_grd_u = np.zeros(self.d)
            for E in self.E_set:
                p_clk_grd_pi[self.pos(E)] += g(set_scores[E])
                p_clk_grd_u += param[self.pos(E)] * dg(set_scores[E]) * set_scores_grd[E] 

            p_clk_grd = p_clk_grd_pi + np.dot(u_bar_grad.T,p_clk_grd_u) 

        # Incorporate p_harm:
        s_hm = sum( [scores[v] for v in self.HM] )
        s_Omega = sum(scores)
        p_hm_val = s_hm/s_Omega * (1.0-p_clk_val)
        
        if grad:
            p_hm_grd_pi = -s_hm/s_Omega * p_clk_grd_pi 
            p_hm_grd_u = (hm_scores_grd_u * s_Omega - s_hm *  omega_scores_grd_u)/(s_Omega**2) * (1-p_clk_val) \
                    - s_hm/s_Omega * p_clk_grd_u
            p_hm_grd = p_hm_grd_pi + np.dot(u_bar_grad.T,p_hm_grd_u) 

        val = -p_clk_val + self.lam * p_hm_val
        if grad:
            grd = -p_clk_grd + self.lam * p_hm_grd

        if self.restart_umap:
            self.set_u_start(u_bar)
        
        if not grad:
            return val
        else:
            return val,grd


class BoundedModelMetrics(Fun):
    def __init__(self,V,E_set,HM,a_HM,a_NH,beta,u0,c=1,eps = 0.00000001,max_iter=100):
        """Parameters are:
            - V: The item matrix V
            - E_set: The set of possible recommendations 
            - HM: The set of harmful items 
            - a_HM,a_NH,beta: the attraction model parameters
            - u0: The inherent profile 
            - c: the recommendation premium parameter
            - lam: the regularization parameter between the two objectives
            - eps: the tolerance of the fixed point convergence
            - max_iter: the maximum number of iterations of the fixed point convergence

        """
        self.V = V
        self.n = V.shape[0] 
        self.d = V.shape[1]
        self.c = c 
        self.HM = HM
        self.eps = eps
        self.max_iter = max_iter
        
        self.s = ScoreGenerator(V)
        self.G = G_Generator(self.n,HM,a_HM,a_NH,beta,u0,V)
        self.p_vE = P_vE_Generator(self.n,E_set,c = self.c)
        self.p = P_Generator(self.p_vE,self.s)
        self.F = F_Generator(self.G,self.p)
        self.u_map = U_MapGenerator(self.F,u0,eps = self.eps, max_iter = self.max_iter) 
        
        self.E_set=sorted(E_set)
        self.E_set_to_index = dict(zip(E_set,range(len(E_set))))
        self.m = len(self.E_set)

    def pos(self,E):
        return self.E_set_to_index[E]

    def set_u_start(self,u_start):
        """ Redefine u_start. Can be useful during gradient descent
        """
        self.u_map = U_MapGenerator(self.F,u_start,eps = self.eps, max_iter = self.max_iter)

    
    def __call__(self,param,lam):
        """Return the loss (and optionally, its gradient) under this given recommendation, as a function of item scores s. """
        # Find fixed point u under input parameters and its gradient/sensitivity
        u_bar, u_bar_grad = self.u_map(param,True)

        # Compute item scores under this user profile
        scores = self.s(u_bar)
    
        # Compute p_click
        def g(s):
            return s/(s+self.c)
        set_scores = {}    
        for E in self.E_set:
            set_scores[E] = sum( [scores[v]  for v in E] )
        p_clk_val = sum( [ param[self.pos(E)] * g(set_scores[E]) for E in self.E_set] )

        # Incorporate p_harm:
        s_hm = sum( [scores[v] for v in self.HM] )
        s_Omega = sum(scores)
        p_hm_val = s_hm/s_Omega * (1.0-p_clk_val)

        val = p_clk_val - lam * p_hm_val
        return val, p_clk_val, p_hm_val


def sampleSet(probs,no_samples=None,rsource = None):
    """Generate sampled sets. 
    """
    n = len(probs)
    d = {}

    assert not (no_samples is None and rsource is None), "no_samples and rsource cannot both be None, at least one must be provided"
    
    if rsource is not None:
        m0,n0 = rsource.shape
        assert n0==n,"Second dimension of rsource should be equal to n=%d, was %d" %(n,n0)
        no_samples = m0
        
    for i in range(no_samples):
        if rsource is not None:
            samples_u = rsource[i]
        else:    
            samples_u = np.random.rand(n)
        E = tuple([ x for x in range(n) if samples_u[x] <= probs[x] ])
        if E in d:
            d[E] += 1
        else:
            d[E] = 1
    E_set = sorted(d.keys())
    weights = np.array([d[E] for E in E_set])/no_samples
    return E_set,weights


class P_SampledGenerator(Fun):
    """A class that generates the selection probabilities  of items given recommendations sampled from a Bernoulli (independent) model.

    """
    def __init__(self,s,c=1,no_samples=None,rsource = None):
        self.no_samples = no_samples
        self.s = s
        self.c = c
        self.rsource = rsource
            

    def __call__(self,param,u,param_grad = False, u_grad=False):
        """Return the probabilities that item is selected (and optionally, its gradient) under this given recommendation, as a function of item scores s. """ 

        E_set,weights= sampleSet(param,no_samples = self.no_samples,rsource = self.rsource)
        #print(len(E_set),len(weights),len(param),sum(weights))

        p_vE = P_vE_Generator(len(param),E_set,c = self.c)
        p = P_Generator(p_vE,self.s)

        
        if u_grad:
            vals,u_grd = p(weights,u,u_grad=True)
        else:
            vals = p(weights,u)

        if param_grad:
            prm_grd = np.zeros((len(vals),len(param)))
            for vi in range(len(param)):
                param_addvi = np.copy(param)
                param_addvi[vi] = 1.0
                #print(param_addvi)
                E_set,weights= sampleSet(param_addvi,no_samples = self.no_samples,rsource = self.rsource)
                #print(len(E_set),sum(weights))
                p_vE = P_vE_Generator(len(param),E_set,c = self.c)
                p = P_Generator(p_vE,self.s)
                vals_add =  p(weights,u)

                param_remvi =np.copy(param)
                param_remvi[vi] = 0.0
                #print(param_remvi)
                E_set,weights= sampleSet(param_remvi,no_samples = self.no_samples,rsource = self.rsource)
                #print(len(E_set),sum(weights))
                p_vE = P_vE_Generator(len(param),E_set,c = self.c)
                p = P_Generator(p_vE,self.s)
                vals_remove = p(weights,u)

                prm_grd[:,vi] = vals_add -vals_remove


        if not (param_grad or u_grad):
            return vals
        else:
            res = [vals]
            if param_grad:
                res.append(prm_grd)
            if u_grad:
                res.append(u_grd)
            return tuple(res)    

    def fix_param(self,param):
        """Return a one-parameter function w.r.t. u, with param fixed.
        """
        return lambda u,grad: self.__call__(param,u,param_grad = False,u_grad=grad)

    def fix_u(self,u):
        """Return a one-parameter function w.r.t. p, with u fixed.
        """
        return lambda param,grad: self.__call__(param,u,param_grad = grad,u_grad=False)


class SampledModelLoss(Fun):
    """A class that generates the loss function under sampled sets.

    """
   
    def __init__(self,V,no_samples,HM,a_HM,a_NH,beta,u0,u0_id,c=1,lam=0,eps=0.01,max_iter=100,fix_rand = True,rsource = None, restart_umap = False):
        """Parameters are:
            - V: The item matrix V
            - no_samples: the number of samples to be used during sampling estimation
            - HM: The set of harmful items 
            - a_HM,a_NH,beta: the attraction model parameters
            - u0: The inherent profile 
            - u0_id: ID of inherent profile
            - c: the recommendation premium parameter
            - lam: the regularization parameter between the two objectives
            - eps: the tolerance of the fixed point convergence
            - max_iter: the maximum number of iterations of the fixed point convergence
            - fix_rand: samples all randomness in the beginning and keeps it fixed from this point on
            - rsource: use pre-computed source of randomness
            - restart_umap: restart the umap function after every function call

        """
        self.V = V
        self.n = V.shape[0] 
        self.d = V.shape[1]
        self.c = c 
        self.HM = HM
        self.lam = lam
        self.no_samples = no_samples
        self.eps =eps
        self.max_iter = max_iter
        self.u0_id = u0_id
        self.restart_umap = restart_umap

        if fix_rand and rsource is None:
            self.rsource = np.random.rand(self.no_samples,self.n)
        else:
            self.rsource = rsource
        
        self.s = ScoreGenerator(self.V)
        self.G = G_Generator(self.n,HM,a_HM,a_NH,beta,u0,V)
        self.p = P_SampledGenerator(self.s,self.c,no_samples = self.no_samples,rsource = self.rsource)

        self.F = F_Generator(self.G,self.p)
        self.u_map = U_MapGenerator(self.F,u0,eps =self.eps,max_iter=self.max_iter) 

    def set_u_start(self,u_start):
        """ Redefine u_start. Can be useful during gradient descent
        """
        self.u_map = U_MapGenerator(self.F,u_start,eps =self.eps,max_iter=self.max_iter)

    def __call__(self,param,grad = False):
        """Return the loss (and optionally, its gradient) under this given recommendation, as a function of item scores s. """
        # print("called Sample Loss")
        # Find fixed point u under input parameters and its gradient/sensitivity
        u_bar, u_bar_grad = self.u_map(param,True)

        # Store u_bar for future use, in case it is needed
        self.u_bar = u_bar
        self.u_bar_grad = u_bar_grad
        
        # Compute item scores under this user profile
        scores = self.s(u_bar)
    
        # Compute p_click and its gradient

        def g(s):
            return s/(s+self.c)
    
        def dg(s):
            return self.c/(s+self.c)**2

        set_scores = {}    

        if grad:
            set_scores_grd = {}
            hm_scores_grd_u = sum(  [scores[v]*self.V[v]  for v in self.HM])
            omega_scores_grd_u = sum(  [scores[v]*self.V[v]  for v in range(self.n)])
 
        E_set,weights = sampleSet(param,no_samples =self.no_samples,rsource = self.rsource)

        for E in E_set:
            set_scores[E] = sum( [scores[v]  for v in E] )
            if grad:
                set_scores_grd[E] =   sum( [scores[v]*self.V[v]  for v in E] )

        p_clk_val = sum( [weights[i] * g(set_scores[E]) for (E,i) in zip(E_set,range(len(weights)))] )
            
        if grad:
            p_clk_grd_u = np.zeros(self.d)
            for E,i in  zip(E_set,range(len(weights))):
                p_clk_grd_u += weights[i] * dg(set_scores[E]) * set_scores_grd[E]  
            
            p_clk_grd_probs = np.zeros(self.n)
            for vi in range(self.n):
                param_addvi = np.copy(param)
                param_addvi[vi] = 1.0

                E_set,weights= sampleSet(param_addvi,self.no_samples)
                set_scores = {}    
                for E in E_set:
                    set_scores[E] = sum( [scores[v]  for v in E] )

                p_clk_val_add = sum( [weights[i] * g(set_scores[E]) for (E,i) in zip(E_set,range(len(weights)))] )

                param_remvi =np.copy(param)
                param_remvi[vi] = 0.0

                E_set,weights= sampleSet(param_remvi,self.no_samples)
                set_scores = {}    
                for E in E_set:
                    set_scores[E] = sum( [scores[v]  for v in E] )

                p_clk_val_rem = sum( [weights[i] * g(set_scores[E]) for (E,i) in zip(E_set,range(len(weights)))] )

                p_clk_grd_probs[vi] =  p_clk_val_add -  p_clk_val_rem

            p_clk_grd = p_clk_grd_probs + np.dot(u_bar_grad.T,p_clk_grd_u) 

        # Incorporate p_harm:
        s_hm = sum( [scores[v] for v in self.HM] )
        s_Omega = sum(scores)
        p_hm_val = s_hm/s_Omega * (1.0-p_clk_val)
        
        if grad:
            p_hm_grd_probs = -s_hm/s_Omega * p_clk_grd_probs
            p_hm_grd_u = (hm_scores_grd_u * s_Omega - s_hm *  omega_scores_grd_u)/(s_Omega**2) * (1-p_clk_val) \
                    - s_hm/s_Omega * p_clk_grd_u
            p_hm_grd = p_hm_grd_probs + np.dot(u_bar_grad.T,p_hm_grd_u) 

        val = -p_clk_val + self.lam * p_hm_val
        if grad:
            grd = -p_clk_grd + self.lam * p_hm_grd

        if self.restart_umap:
            self.set_u_start(u_bar)
        
        if not grad:
            return val
        else:
            return val,grd


class SampledModelMetrics(Fun):
    """
    generates obj, pCLK, pH values for a given policy
    """
    def __init__(self,V,no_samples,HM,a_HM,a_NH,beta,u0,c=1,eps=0.01,max_iter=100,fix_rand = True,rsource = None):
        """Parameters are:
            - V: The item matrix V
            - no_samples: the number of samples to be used during sampling estimation
            - HM: The set of harmful items 
            - a_HM,a_NH,beta: the attraction model parameters
            - u0: The inherent profile 
            - c: the recommendation premium parameter
            - lam: the regularization parameter between the two objectives
            - eps: the tolerance of the fixed point convergence
            - max_iter: the maximum number of iterations of the fixed point convergence
            - fix_rand: samples all randomness in the beginning and keeps it fixed from this point on
            - rsource: use pre-computed source of randomness

        """
        self.V = V
        self.n = V.shape[0] 
        self.d = V.shape[1]
        self.c = c 
        self.HM = HM
        self.no_samples = no_samples
        self.eps =eps
        self.max_iter = max_iter
        

        if fix_rand and rsource is None:
            self.rsource = np.random.rand(self.no_samples,self.n)
        else:
            self.rsource = rsource

        
        self.s = ScoreGenerator(V)
        self.G = G_Generator(self.n,HM,a_HM,a_NH,beta,u0,V)
        self.p = P_SampledGenerator(self.s,self.c,no_samples = self.no_samples,rsource = self.rsource)


        self.F = F_Generator(self.G,self.p)
        self.u_map = U_MapGenerator(self.F,u0,eps =self.eps,max_iter=self.max_iter) 

    def __call__(self, param, lam):
        "returns obj, pCLK, pH for given policy probability distribution over items"

        # Find fixed point u under input parameters and its gradient/sensitivity
        u_bar = self.u_map(param,False)

        # Compute item scores under this user profile
        scores = self.s(u_bar)
    
        # Compute p_click
        def g(s):
            return s/(s+self.c)
        set_scores = {}    
        E_set,weights = sampleSet(param,self.no_samples,self.rsource)
        for E in E_set:
            set_scores[E] = sum( [scores[v] for v in E] )
        p_clk_val = sum( [weights[i] * g(set_scores[E]) for (E,i) in zip(E_set,range(len(weights)))] )

        # Incorporate p_harm:
        s_hm = sum( [scores[v] for v in self.HM] )
        s_Omega = sum(scores)
        p_hm_val = s_hm/s_Omega * (1.0-p_clk_val)

        # Final Objective
        val = p_clk_val - lam * p_hm_val
        return val, p_clk_val, p_hm_val



### This code is not necessary. use self.umap of a Loss object instead.
class SampledModelMap(Fun):
    """
    computes u_bar, scores
    """
    def __init__(self,V,no_samples,HM,a_HM,a_NH,beta,u0,c=1,eps=0.01,max_iter=100,fix_rand = True,rsource = None):
        """Parameters are:
            - V: The item matrix V
            - no_samples: the number of samples to be used during sampling estimation
            - HM: The set of harmful items 
            - a_HM,a_NH,beta: the attraction model parameters
            - u0: The inherent profile 
            - c: the recommendation premium parameter
            - lam: the regularization parameter between the two objectives
            - eps: the tolerance of the fixed point convergence
            - max_iter: the maximum number of iterations of the fixed point convergence
            - fix_rand: samples all randomness in the beginning and keeps it fixed from this point on
            - rsource: use pre-computed source of randomness

        """
        self.V = V
        self.n = V.shape[0] 
        self.d = V.shape[1]
        self.c = c 
        self.HM = HM
        self.no_samples = no_samples
        self.eps =eps
        self.max_iter = max_iter
        


        if fix_rand and rsource is None:
            self.rsource = np.random.rand(self.no_samples,self.n)
        else:
            self.rsource = rsource

        
        self.s = ScoreGenerator(V)
        self.G = G_Generator(self.n,HM,a_HM,a_NH,beta,u0,V)
        self.p = P_SampledGenerator(self.s,self.c,no_samples = self.no_samples,rsource = self.rsource)


        self.F = F_Generator(self.G,self.p)
        self.u_map = U_MapGenerator(self.F,u0,eps =self.eps,max_iter=self.max_iter) 


    def __call__(self, param):
        "returns obj, pCLK, pH for given policy probability distribution over items"

        # Find fixed point u under input parameters and its gradient/sensitivity
        u_bar = self.u_map(param,False)

        # Compute item scores under this user profile
        scores = self.s(u_bar)

        return u_bar, scores

    def set_u_start(self,u_start):
        """ Redefine u_start. Can be useful during gradient descent
        """
        self.u_map = U_MapGenerator(self.F,u_start,eps =self.eps,max_iter=self.max_iter)
    






if __name__=="__main__":
    #d = 10
    #n = 200
    d = 3
    n = 10
    k = 3
    # eps = 0.0000001
    eps = 0.1
    delta = 0.1
    E_set = list(itertools.combinations(range(n-2),k))
    m = len(E_set)
    HM = [n-2,n-1]

    # no_samples = 100000
    no_samples = 1000

    
    # for i in range(100):
    for i in range(2):

        # print("---BOUNDED CARDINALITY TEST %d---" %i)
        c=3*np.random.rand()
        
        V = np.random.randn(n,d)
        u = np.random.randn(d)

        #s = ScoreGenerator(V)
        # gradTester(s,u,eps)

        #p_vE = P_vE_Generator(n,E_set,c)

        # s_val = s(u)
        # gradTester(p_vE,s_val)   

        #p = P_Generator(p_vE,s)
        

        # param = np.random.rand(m)
        # param = param/param.sum()
        
        # p_u = p.fix_u(u) 
        # p_param = p.fix_param(param)
        
        # gradTester(p_param,u)
        # gradTester(p_u,param)

        # p_val = p(param,u)
        
        r = np.random.rand(3) 
        a_HM,a_NH,beta = r/r.sum()
        u0 = np.random.randn(d)

        G = G_Generator(n,HM,a_HM,a_NH,beta,u0,V)

        # gradTester(G,p_val)

        #F = F_Generator(G,p)
        # F_u = F.fix_u(u) 
        # F_param = F.fix_param(param)

        # gradTester(F_param,u)
        # gradTester(F_u,param)


        # u_start = np.random.randn(d)

        # u_map = U_MapGenerator(F,u_start)

        # gradTester(u_map,param)
        


        lam = 100**np.random.rand()
        # loss = BoundedCardinalityModelLoss(V,E_set,HM,a_HM,a_NH,beta,u0,c,lam)

        # gradTester(loss,param)

        print("---SAMPLED SET TEST %d---" %i)
          
        loss = SampledModelLoss(V,no_samples,HM,a_HM,a_NH,beta,u0,"user",c,lam)
        
        sample_p = loss.p

        probs = np.random.rand(n)
        probs[-1]=0
        probs[-2]=0
        probs = probs/probs.sum() * k

        
        sample_p_u = sample_p.fix_u(u) 
        sample_p_param = sample_p.fix_param(probs)
        
        gradTester(sample_p_param,u,delta=delta,eps=eps)
        gradTester(sample_p_u,probs,delta=delta,eps=eps)

        F = loss.F
        F_u = F.fix_u(u) 
        F_param = F.fix_param(probs)

        gradTester(F_param,u,eps=eps,delta=delta)
        gradTester(F_u,probs,eps=eps,delta=delta)


        u_start = np.random.randn(d)

        # u_map = U_MapGenerator(F,u_start,eps = 0.001,max_iter=200)
        u_map = loss.u_map

        gradTester(u_map,probs,eps=eps,delta=delta)
        
        gradTester(loss,probs,eps=eps,delta = delta)

       

