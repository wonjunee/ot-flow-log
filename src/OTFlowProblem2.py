# OTFlowProblem.py
import math
import torch
import numpy as np
from torch.nn.functional import pad
from src.Phi import *


def vec(x):
    """vectorize torch tensor x"""
    return x.view(-1,1)


def W(x):
    normx2 = (x).square().sum(dim = 2)
    print("Wx", normx2/2 - (normx2.sqrt() + 0.001).log())
    return normx2/2 - (normx2.sqrt() + 0.1).log()
   

def get_samples_next_time_step(x, Phi, nt, stepper="rk4", alph =[1.0,1.0,1.0] ):
    h = 1.0 / nt

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 3, 0, 0), value=0)

    # get the inital density from net_list. Given an initial density at t=0, iterate through n_tau - 1
    with torch.no_grad(): 
        tk = 0

        # given the data from the list of nets, compute z for this iteration
        if stepper=='rk4':
            for k in range(nt):
                z = stepRK4(odefun, z, Phi, alph, tk, tk + h)
                tk += h
        elif stepper=='rk1':
            for k in range(nt):
                z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
                tk += h

        d = z.shape[1]-3 # dimension for x
        print("shape: ", z[:,d].shape)
    return z[:,:d]

def get_samples_next_time_step_including_det(x, rho, net, net_list, nt, n_tau, stepper="rk4", alph =[1.0,1.0,1.0] ):
    h = 1.0 / nt

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 3, 0, 0), value=0)

    d = z.shape[1]-3 # dimension for x

    # get the inital density from net_list. Given an initial density at t=0, iterate through n_tau - 1
    rho_next = rho
    with torch.no_grad(): 
        for n in range(n_tau):
            tk = 0
            for k in range(nt):
                z = stepRK4_nonlinear(odefun_nonlinear, z, net_list[n], rho_next, alph, tk, tk + h)
                tk += h
                rho_next = rho_next / torch.exp(z[:,d])

            z = pad(z[:,0:d], (0,3,0,0), value=0)
        tk = 0
        for k in range(nt):
            z = stepRK4_nonlinear(odefun_nonlinear, z, net, rho_next,  alph, tk, tk + h)
            tk += h
    # z[:,d] = log det
    return z[:,:d], rho_next


def OTFlowProblemGradientFlows(x, Phi, tspan , nt, n_tau, net_list, stepper="rk4", alph =[1.0,1.0,1.0] ):
    """

    Evaluate objective function of OT Flow problem; see Eq. (8) in the paper.
    
    :param x:       input data tensor nex-by-d
    :param Phi:     neural network
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param n_tau: nth step in gradient flows
    :param net_list: list of nets of length n_tau
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :return:
        Jc - float, objective function value dot(alph,cs)
        cs - list length 5, the five computed costs
    """
    h = (tspan[1]-tspan[0]) / nt

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 3, 0, 0), value=0)

    # get the inital density from net_list. Given an initial density at t=0, iterate through n_tau - 1
    # with torch.no_grad(): 
    #     for n in range(n_tau):
    #         tk = tspan[0]
    #         if stepper=='rk4':
    #             for k in range(nt):
    #                 z = stepRK4(odefun, z, net_list[n], alph, tk, tk + h)
    #                 tk += h
    #         elif stepper=='rk1':
    #             for k in range(nt):
    #                 z = stepRK1(odefun, z, net_list[n], alph, tk, tk + h)
    #                 tk += h

    #         z = pad(z[:,0:z.shape[1]-3], (0,3,0,0), value=0)


    tk = tspan[0]

    # given the data from the list of nets, compute z for this iteration
    if stepper=='rk4':
        for k in range(nt):
            z = stepRK4(odefun, z, Phi, alph, tk, tk + h)
            tk += h
    elif stepper=='rk1':
        for k in range(nt):
            z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
            tk += h

    d = z.shape[1]-3 # dimension for x

    # terminal_cost =  -( torch.sum(  -0.5 * math.log(2*math.pi) - torch.pow(z[:,0:d] - 3,2) / 2  , 1 , keepdims=True ) ).mean()

    # interaction cost
    n  = z.shape[0]
    Tx = z[:,0:d]

    normx2 = (Tx.view((n, 1, d)) - Tx.view((1, n, d))).square().mean(dim = 2)
    normx4 = normx2.square()

    # terminal_cost = (normx2 - (normx2 + 1e-8).log()).mean()/2
    terminal_cost = (normx4/4 - normx2/2).mean()


    # terminal_cost =  -( torch.sum( - torch.pow(z[:,0:d] - 3,2) / 2  , 1 , keepdims=True ) ).mean()
    # terminal_cost =  -( torch.sum(  -0.5 * math.log(2*math.pi) + torch.log( torch.exp(- torch.pow(z[:,0:d] - 3,2) / 2) + torch.exp(- torch.pow(z[:,0:d] + 3,2) / 2))   , 1 , keepdims=True ) ).mean()
    # ASSUME all examples are equally weighted

    # torch.mean(C(z)): log det cost
    costL  = torch.mean(z[:,-2])
    # costC  = terminal_cost + torch.mean(C(z))
    tau = 0.01
    costC  = terminal_cost * tau
    # costR  = torch.mean(z[:,-1])
    costR = 0

    cs = [costL, costC, costR]
    return sum(i[0] * i[1] for i in zip(cs, alph)) , cs

def multiple_exp(Tx):
    # return 1/(np.sqrt(2*np.pi)) * torch.exp(-(Tx-1).square()/(2 * 0.5*0.5)) + torch.exp(- (Tx).square()/(2 * 0.5 *0.5)) + torch.exp(-(Tx+1.5).square()/(2*0.5 *0.5))
    return 1/(np.sqrt(2*np.pi)) * ( torch.exp(-(Tx-torch.tensor([[2.0,-2.0]])).square()/(2 * 0.5**2)) + torch.exp(-(Tx-torch.tensor([[-2.0,2.0]])).square()/(2 * 0.5**2)) )

def compute_U(rho_next, Tx=None):
    # return rho_next.square()
    # return rho_next.square() + 0.1 * rho_next * ( torch.log(rho_next) )
    if Tx != None:
        return rho_next * ( torch.log(rho_next) - torch.log(multiple_exp(Tx) ).mean() - 1 )
    else:
        return rho_next * ( torch.log(rho_next) - 1 )

def OTFlowPorous(x, rho, Phi, tspan , nt, n_tau, net_list, stepper="rk4", alph =[1.0,1.0,1.0], z=None):
    """

    Evaluate objective function of OT Flow problem; see Eq. (8) in the paper.
    
    :param x:       input data tensor nex-by-d
    :param rho:     input rho nex-by-1
    :param Phi:     neural network
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param n_tau: nth step in gradient flows
    :param net_list: list of nets of length n_tau
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :return:
        Jc - float, objective function value dot(alph,cs)
        cs - list length 5, the five computed costs
    """
    h = (tspan[1]-tspan[0]) / nt

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    # z = pad(x, (0, 3, 0, 0), value=0)

    d = z.shape[1]-3 # dimension for x

    # get the inital density from net_list. Given an initial density at t=0, iterate through n_tau - 1
    rho_next = rho
    # with torch.no_grad(): 
    #     for n in range(n_tau):
    #         tk = tspan[0]
    #         for k in range(nt):
    #             z = stepRK4(odefun, z, net_list[n], alph, tk, tk + h)
    #             tk += h
    #             rho_next = rho_next / torch.exp(z[:,d])

    #         z = pad(z[:,0:d], (0,3,0,0), value=0)

    tk = tspan[0]

    # given the data from the list of nets, compute z for this iteration
    # if stepper=='rk4':
    #     for k in range(nt):
    #         z = stepRK4(odefun, z, Phi, alph, tk, tk + h)
    #         tk += h
    # elif stepper=='rk1':
    #     for k in range(nt):
    #         z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
    #         tk += h
    for k in range(nt):
        z = stepRK4_nonlinear(odefun_nonlinear, z, Phi, rho, alph, tk, tk + h)
        tk += h

    # terminal_cost =  -( torch.sum(  -0.5 * math.log(2*math.pi) - torch.pow(z[:,0:d] - 3,2) / 2  , 1 , keepdims=True ) ).mean()

    # interaction cost
    n  = z.shape[0]
    Tx = z[:,0:d]

    # normx4 = normx2.square()

    # terminal_cost = (normx2 - (normx2 + 1e-8).log()).mean()/2
    # terminal_cost = (normx4/4 - normx2/2).mean()
    
    # with torch.no_grad():
    #     print("z:", torch.max(z[:,d].exp()), torch.min(z[:,d].exp()))

    # rho_next = rho_next / torch.exp(z[:,d])
    rho_next = torch.exp(torch.log(rho) - z[:,d] )

    cost1 = (compute_U(rho_next,Tx) / rho_next).mean()

    terminal_cost = cost1
    # terminal_cost =  -( torch.sum(  -0.5 * math.log(2*math.pi) + torch.log( torch.exp(- torch.pow(z[:,0:d] - 3,2) / 2) + torch.exp(- torch.pow(z[:,0:d] + 3,2) / 2))   , 1 , keepdims=True ) ).mean()
    # ASSUME all examples are equally weighted

    # torch.mean(C(z)): log det cost
    costL  = torch.mean(z[:,-2])
    # costC  = terminal_cost + torch.mean(C(z))
    # tau = 0.15
    tau = 0.01
    costC  = terminal_cost * tau
    # costR  = torch.mean(z[:,-1])
    costR = 0

    cs = [costL, costC, costR]
    return sum(i[0] * i[1] for i in zip(cs, alph)) , cs

def OTFlowProblem(x, Phi, tspan , nt, stepper="rk4", alph =[1.0,1.0,1.0] ):
    """

    Evaluate objective function of OT Flow problem; see Eq. (8) in the paper.

    :param x:       input data tensor nex-by-d
    :param Phi:     neural network
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :return:
        Jc - float, objective function value dot(alph,cs)
        cs - list length 5, the five computed costs
    """
    h = (tspan[1]-tspan[0]) / nt

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 3, 0, 0), value=0)

    tk = tspan[0]

    if stepper=='rk4':
        for k in range(nt):
            z = stepRK4(odefun, z, Phi, alph, tk, tk + h)
            tk += h
    elif stepper=='rk1':
        for k in range(nt):
            z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
            tk += h

    d = z.shape[1]-3 # dimension for x

    # terminal_cost =  -( torch.sum(  -0.5 * math.log(2*math.pi) - torch.pow(z[:,0:d] - 3,2) / 2  , 1 , keepdims=True ) ).mean()

    # interaction cost
    n  = z.shape[0]
    Tx = z[:,0:d]

    normx2 = (Tx.view((n, 1, d)) - Tx.view((1, n, d))).square().mean(dim = 2)
    # print((normx2.sqrt() + 1e-6))
    terminal_cost = (normx2 - (normx2 + 1e-8).log()).mean()/2
    # terminal_cost = normx2.mean()/2


    # terminal_cost =  -( torch.sum( - torch.pow(z[:,0:d] - 3,2) / 2  , 1 , keepdims=True ) ).mean()
    # terminal_cost =  -( torch.sum(  -0.5 * math.log(2*math.pi) + torch.log( torch.exp(- torch.pow(z[:,0:d] - 3,2) / 2) + torch.exp(- torch.pow(z[:,0:d] + 3,2) / 2))   , 1 , keepdims=True ) ).mean()
    # ASSUME all examples are equally weighted

    # torch.mean(C(z)): log det cost
    costL  = torch.mean(z[:,-2])
    # costC  = terminal_cost + torch.mean(C(z))
    tau = 0.1
    costC  = terminal_cost  * tau
    # costR  = torch.mean(z[:,-1])
    costR = 0

    cs = [costL, costC, costR]

    # return dot(cs, alph)  , cs
    return sum(i[0] * i[1] for i in zip(cs, alph)) , cs



def stepRK4(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 4 integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the OT-Flow Problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """

    h = t1 - t0 # step size
    z0 = z

    K = h * odefun(z0, t0, Phi, alph=alph)
    z = z0 + (1.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + K , t0+h , Phi, alph=alph)
    z += (1.0/6.0) * K

    return z



def stepRK4_nonlinear(odefun, z, Phi, rho, alph, t0, t1):
    """
        Runge-Kutta 4 integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the OT-Flow Problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """

    h = t1 - t0 # step size
    z0 = z

    K = h * odefun(z0, t0, Phi, rho, alph=alph)
    z = z0 + (1.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, rho, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, rho, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + K , t0+h , Phi, rho, alph=alph)
    z += (1.0/6.0) * K

    return z

def stepRK1(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 1 / Forward Euler integration scheme.  Added for comparison, but we recommend stepRK4.
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the mean field game problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """
    z0 = z
    z += (t1 - t0) * odefun(z0, t0, Phi, alph=alph)
    return z

def integrate_gf(x, net, tspan , nt, n_tau, net_list, stepper="rk4", alph =[1.0,1.0,1.0], intermediates=False ):
    """
        perform the time integration in the d-dimensional space
    :param x:       input data tensor nex-by-d
    :param net:     neural network Phi
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :param intermediates: bool, True means save all intermediate time points along trajectories
    :return:
        z - tensor nex-by-d+4, features at time t1
        OR zFull - tensor nex-by-d+3-by-nt+1 , trajectories from time t0 to t1 (when intermediates=True)
    """

    h = (tspan[1]-tspan[0]) / nt

    # initialize "hidden" vector to propagate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 3, 0, 0), value=tspan[0])

    # get the inital density from net_list. Given an initial density at t=0, iterate through n_tau - 1
    with torch.no_grad(): 
        for n in range(n_tau):
            tk = tspan[0]
            if stepper=='rk4':
                for k in range(nt):
                    z = stepRK4(odefun, z, net_list[n], alph, tk, tk + h)
                    tk += h
            elif stepper=='rk1':
                for k in range(nt):
                    z = stepRK1(odefun, z, net_list[n], alph, tk, tk + h)
                    tk += h

            z = pad(z[:,0:z.shape[1]-3], (0,3,0,0), value=0)  
    
    tk = tspan[0]

    if stepper == 'rk4':
        for k in range(nt):
            z = stepRK4(odefun,z,net, alph,tk,tk+h)
            tk += h
    elif stepper == 'rk1':
        for k in range(nt):
            z = stepRK1(odefun,z,net, alph,tk,tk+h)
            tk += h

    return z



def integrate(x, net, tspan , nt, stepper="rk4", alph =[1.0,1.0,1.0], intermediates=False ):
    """
        perform the time integration in the d-dimensional space
    :param x:       input data tensor nex-by-d
    :param net:     neural network Phi
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :param intermediates: bool, True means save all intermediate time points along trajectories
    :return:
        z - tensor nex-by-d+4, features at time t1
        OR zFull - tensor nex-by-d+3-by-nt+1 , trajectories from time t0 to t1 (when intermediates=True)
    """

    h = (tspan[1]-tspan[0]) / nt

    # initialize "hidden" vector to propagate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 3, 0, 0), value=tspan[0])

    tk = tspan[0]

    if intermediates: # save the intermediate values as well
        zFull = torch.zeros( *z.shape , nt+1, device=x.device, dtype=x.dtype) # make tensor of size z.shape[0], z.shape[1], nt
        zFull[:,:,0] = z

        if stepper == 'rk4':
            for k in range(nt):
                zFull[:,:,k+1] = stepRK4(odefun, zFull[:,:,k] , net, alph, tk, tk+h)
                tk += h
        elif stepper == 'rk1':
            for k in range(nt):
                zFull[:,:,k+1] = stepRK1(odefun, zFull[:,:,k] , net, alph, tk, tk+h)
                tk += h

        return zFull

    else:
        if stepper == 'rk4':
            for k in range(nt):
                z = stepRK4(odefun,z,net, alph,tk,tk+h)
                tk += h
        elif stepper == 'rk1':
            for k in range(nt):
                z = stepRK1(odefun,z,net, alph,tk,tk+h)
                tk += h

        return z

    # return in case of error
    return -1



def C(z):
    """Expected negative log-likelihood; see Eq.(3) in the paper"""
    d = z.shape[1]-3
    l = z[:,d] # log-det

    return -( torch.sum(  -0.5 * math.log(2*math.pi) - torch.pow(z[:,0:d],2) / 2  , 1 , keepdims=True ) + l.unsqueeze(1) )


def odefun(x, t, net, alph=[1.0,1.0,1.0]):
    """
    neural ODE combining the characteristics and log-determinant (see Eq. (2)), the transport costs (see Eq. (5)), and
    the HJB regularizer (see Eq. (7)).

    d_t  [x ; l ; v ; r] = odefun( [x ; l ; v ; r] , t )

    x - particle position
    l - log determinant
    v - accumulated transport costs (Lagrangian)
    r - accumulates violation of HJB condition along trajectory
    """
    nex, d_extra = x.shape
    d = d_extra - 3

    z = pad(x[:, :d], (0, 1, 0, 0), value=t) # concatenate with the time t

    # print("zshape:", z.shape, "d", d)

    gradPhi, trH = net.trHess(z)

    dx = -(1.0/alph[0]) * gradPhi[:,0:d]
    dl = -(1.0/alph[0]) * trH.unsqueeze(1)
    dv = 0.5 * torch.sum(torch.pow(dx, 2) , 1 ,keepdims=True)
    dr = torch.abs(  -gradPhi[:,-1].unsqueeze(1) + alph[0] * dv  ) 
    
    return torch.cat( (dx,dl,dv,dr) , 1  )



def odefun_nonlinear(x, t, net, rho, alph=[1.0,1.0,1.0]):
    """
    neural ODE combining the characteristics and log-determinant (see Eq. (2)), the transport costs (see Eq. (5)), and
    the HJB regularizer (see Eq. (7)).

    d_t  [x ; l ; v ; r] = odefun( [x ; l ; v ; r] , t )

    x - particle position
    l - log determinant
    v - accumulated transport costs (Lagrangian)
    r - accumulates violation of HJB condition along trajectory
    """
    nex, d_extra = x.shape
    d = d_extra - 3

    z = pad(x[:, :d], (0, 1, 0, 0), value=t) # concatenate with the time t

    # print("zshape:", z.shape, "d", d)

    gradPhi, trH = net.trHess(z)

    dx = -(1.0/alph[0]) * gradPhi[:,0:d]
    dl = -(1.0/alph[0]) * trH.unsqueeze(1)
    rho_next = torch.exp(torch.log(rho) - z[:,d])
    # dv = 0.5 * torch.sum(torch.pow(dx, 2) * (1-rho_next.view((rho_next.shape[0],1))) , 1 ,keepdims=True)
    dv = 0.5 * torch.sum(torch.pow(dx, 2) / (1-rho_next.view((rho_next.shape[0],1))) , 1 ,keepdims=True)
    dr = torch.abs(  -gradPhi[:,-1].unsqueeze(1) + alph[0] * dv  ) 
    
    return torch.cat( (dx,dl,dv,dr) , 1  )


