# trainToyOTflow.py
# training driver for the two-dimensional toy problems
import argparse
import os
import time
import datetime
import torch.optim as optim
import numpy as np
import math
import lib.toy_data as toy_data
import lib.utils as utils
from lib.utils import count_parameters
from src.plotter import plot4_gf
from src.OTFlowProblem import *
import config
import matplotlib.pyplot as plt

cf = config.getconfig()

parser = argparse.ArgumentParser('OT-Flow')
parser.add_argument(
    '--data', choices=['square', 'porous', 'swissroll', '8gaussians','2gaussians', '1gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='8gaussians'
)

parser.add_argument("--nt"    , type=int, default=1, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=1, help="number of time steps for validation")
parser.add_argument("--tau", type=float, default=0.01, help="the time stepsize for the outer JKO iteration")
parser.add_argument('--alph'  , type=str, default='1.0,1.0,0.0')
parser.add_argument('--m'     , type=int, default=64)
parser.add_argument('--nTh'   , type=int, default=3)

parser.add_argument('--niters'        , type=int  , default=4000)
parser.add_argument('--batch_size'    , type=int  , default=1000)
parser.add_argument('--val_batch_size', type=int  , default=1000)

parser.add_argument('--lr'          , type=float, default=0.0001)
parser.add_argument("--drop_freq"   , type=int  , default=500, help="how often to decrease learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr_drop'     , type=float, default=1.00001)
parser.add_argument('--optim'       , type=str  , default='adam', choices=['adam'])
parser.add_argument('--prec'        , type=str  , default='single', choices=['single','double'], help="single or double precision")

parser.add_argument('--save'    , type=str, default='experiments')
parser.add_argument('--viz_freq', type=int, default=200)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--gpu'     , type=int, default=0)
parser.add_argument('--sample_freq', type=int, default=25)

parser.add_argument('--n_tau', type=int, default=0)

args = parser.parse_args()

args.alph = [float(item) for item in args.alph.split(',')]

# get precision type
if args.prec =='double':
    prec = torch.float64
else:
    prec = torch.float32

# get timestamp for saving models
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

save_folder = args.save
os.makedirs(save_folder, exist_ok=True)
os.makedirs(f"{args.save}/figs", exist_ok=True)

# logger
utils.makedirs(args.save)
# logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
# print("start time: " + start_time)
# print(args)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')



def compute_loss(net, x, nt):
    Jc , cs = OTFlowProblem(x, net, [0,1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs

def compute_loss_gf(net, x, nt, tau, n_tau, net_list, rho, z):
    Jc , cs = OTFlowProblemGradientFlowsPorous(x, rho, net, [0,1], nt=nt, tau=tau, n_tau=n_tau, net_list=net_list, stepper="rk4", alph=net.alph, z=z)
    return Jc, cs

def plot_scatter(x_next, args, n_tau, sPath):
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    x_next = x_next.numpy()
    ax.scatter(x_next[:,0], x_next[:,1], marker='.')
    ax.set_xlim([-4.1,4.1])
    ax.set_ylim([-4.1,4.1])
    plt.tight_layout()
    ax.set_aspect(True)
    plt.savefig(sPath)
    plt.close('all')

def plot_scatter_color(x_next, args, n_tau, color, sPath):
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    x_next = x_next.numpy()
    ax.scatter(x_next[:,0], x_next[:,1], c=color, marker='.')
    ax.set_xlim([-4.1,4.1])
    ax.set_ylim([-4.1,4.1])
    plt.tight_layout()
    ax.set_aspect(True)
    plt.savefig(sPath)
    plt.close('all')

def plot_scatter_color_2(x_next, args, n_tau, color, color2, sPath):
    fig = plt.figure(figsize=(8,4))
    x_next = x_next.numpy()
    ax = fig.add_subplot(121)
    ax.scatter(x_next[:,0], x_next[:,1], c=color, marker='.')
    ax.set_xlim([-4.1,4.1])
    ax.set_ylim([-4.1,4.1])
    plt.tight_layout()
    ax.set_aspect(True)
    ax = fig.add_subplot(122)
    ax.scatter(x_next[:,0], x_next[:,1], c=color2, marker='.')
    ax.set_xlim([-4.1,4.1])
    ax.set_ylim([-4.1,4.1])
    plt.tight_layout()
    ax.set_aspect(True)
    plt.savefig(sPath)
    plt.close('all')

if __name__ == '__main__':

    torch.set_default_dtype(prec)
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    # neural network for the potential function Phi
    d      = 2
    alph   = args.alph
    nt     = args.nt
    nt_val = args.nt_val
    tau    = args.tau
    nTh    = args.nTh
    m      = args.m
    net = Phi(nTh=nTh, m=args.m, d=d, alph=alph)
    net = net.to(prec).to(device)
    n_tau = args.n_tau


    b1 = 0.5
    b2 = 0.999
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(b1, b2))
    net_list = []

    for i in range(n_tau):
        filename = os.path.join(args.save, '{:}_alph{:}_{:}_m{:}_n{}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m,i))
        print(f"loading {filename}")
        checkpt  = torch.load(filename, map_location=lambda storage, loc: storage)
        net1     = Phi(nTh=nTh, m=m, d=d, alph=alph)
        prec1    = checkpt['state_dict']['A'].dtype
        net1     = net1.to(prec1).to(device)
        net1.load_state_dict(checkpt['state_dict'])
        net_list.append(net1)

        if i == n_tau - 1:
            net.load_state_dict(checkpt['state_dict'])
            x0 = checkpt['x0']
            rho = checkpt['rho']
            energy_list = checkpt['energy']
            energy_list.append(0)
            error_list = checkpt['error']
            error_list.append(0)


    # if it is the first outer iteration then sample random points
    if n_tau == 0:
        x0,rho = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
        print(x0.shape, rho.shape)
        x0 = cvt(torch.from_numpy(x0))
        rho = cvt(torch.from_numpy(rho))
        energy_list = [0]
        error_list = [0]

        # for plotting
        x0plot,rhoplot = toy_data.inf_train_gen(args.data, batch_size=1000)
        x0plot = cvt(torch.from_numpy(x0plot))
        rhoplot = cvt(torch.from_numpy(rhoplot))
        plot_scatter_color(x0plot, args, n_tau, rhoplot, sPath = os.path.join(args.save, 'figs', f"rho_{n_tau}.png"))
        del x0plot
        del rhoplot

        # args.niters    = 1000
    # else choose the same points from the previous outer iteration which is happening in previous lines.

    # print(net)
    print("-------------------------")
    print("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d,m,nTh,alph))
    print("nt={:}   nt_val={:}   tau={:}  n_tau={:}".format(nt,nt_val,tau,n_tau))
    print("Number of trainable parameters: {}".format(count_parameters(net)))
    print("-------------------------")
    print(str(optim)) # optimizer info
    print("data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu))
    print("maxIters={:} val_freq={:} viz_freq={:}".format(args.niters, args.val_freq, args.viz_freq))
    print("saveLocation = {:}".format(args.save))
    print("-------------------------\n")

    end = time.time()
    best_loss = float('inf')
    bestParams = None

    # setup data [nSamples, d]
    # use one batch as the entire data set


    x0val,rhoval = toy_data.inf_train_gen(args.data, batch_size=args.val_batch_size)
    x0val = cvt(torch.from_numpy(x0val))

    log_msg = (
        '{:5s}  {:6s}   {:9s}  {:9s}  {:9s}  {:9s}  {:9s}      {:9s}  {:9s}  {:9s}  {:9s}  '.format(
            'iter', ' time','loss', 'L (L_2)', 'C (loss)', 'R (HJB)', 'rel loss', 'valLoss', 'valL', 'valC', 'valR'
        )
    )
    print(log_msg)

    time_meter = utils.AverageMeter()

    net.train()

    previous_loss = 1

    ccc = True # for plotting in between

    global_loss = 1

    best_params = net.state_dict()

    for itr in range(0, args.niters + 1):
        # train sampling
        if itr % 1 == 0:
            x0,rho = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
            x0 = cvt(torch.from_numpy(x0))
            rho = cvt(torch.from_numpy(rho))

            h = 1.0 / nt

            # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
            # z = pad(x0, (0, 3, 0, 0), value=0)
            z = torch.zeros((x0.shape[0], d+3)) ; z[:,:d] = x0
            # get the inital density from net_list. Given an initial density at t=0, iterate through n_tau - 1
            rho_next = rho
            with torch.no_grad(): 
                for n in range(n_tau):
                    tk = 0
                    for k in range(nt):
                        z = stepRK4(odefun, z, net_list[n], alph, tk, tk + h)
                        tk += h
                        rho_next = rho_next / torch.exp(z[:,d])
                    # z = pad(z[:,0:d], (0,3,0,0), value=0)
                    z[:,d:] = 0

        optim.zero_grad()
        loss, costs  = compute_loss_gf(net, x0, nt=nt, tau=tau, n_tau=n_tau, net_list=net_list, rho=rho, z=z)
        loss.backward()
        optim.step()

        time_meter.update(time.time() - end)

        # stopping condition
        tol = 1e-10
        rel_loss = abs(previous_loss - loss)/abs(previous_loss)
        # if rel_loss < tol or itr == args.niters:
        if itr == args.niters:
            # best_loss   = test_loss.item()
            # best_costs = test_costs
            with torch.no_grad():
                best_loss   = loss.item()
                best_costs  = costs
                utils.makedirs(args.save)
                best_params = net.state_dict()

                x0val,rhoval = toy_data.inf_train_gen(args.data, batch_size=1000)
                x0val  = cvt(torch.from_numpy(x0val))
                rhoval = cvt(torch.from_numpy(rhoval))

                # compute samples at the next time step
                x_next, rho_next = get_samples_next_time_step_including_det(x0val, rhoval, net, net_list, nt, n_tau)
                # plot_scatter(x_next, args, n_tau, sPath = os.path.join(args.save, 'figs', 'plot_n_{:04d}.png'.format(n_tau+1)))

                mean = torch.mean(x_next, dim=0)
                var  = torch.var(x_next)

                print(f"mean = {mean} variance = {var}")
                plot_scatter_color(x_next, args, n_tau, rho_next, sPath = os.path.join(args.save, 'figs', f"rho_{n_tau+1}.png"))

                t0  = 0.001
                tau = 0.005
                t = tau * (n_tau+1)

                m_power = 2
                A = (4 * np.pi * m_power * (t+t0)) ** ((1-m_power)/m_power)
                B = (m_power-1)/(4*m_power**2*(t+t0))
                rho_next_exact = np.zeros((x_next.shape[0]))
                for i,x in enumerate(x_next):
                    xnorm2 = x[0]**2 + x[1]**2
                    rho_next_exact[i] = max(A - B * xnorm2, 0)**(1/(m_power-1))
                plot_scatter_color_2(x_next, args, n_tau, rho_next, rho_next_exact, sPath = os.path.join(args.save, 'figs', f"rho2_{n_tau+1}.png"))

                rho_next = rho_next.numpy()
                energy_list[-1] = costs[1]
                # error_list[-1]  = np.mean((rho_next/np.max(rho_next) - rho_next_exact/np.max(rho_next_exact))**2)
                error_list[-1]  = np.mean((rho_next - rho_next_exact)**2)
                np.savetxt("error.csv", np.array(error_list), delimiter=",")

                torch.save({
                    'args': args,
                    'state_dict': best_params,
                    'x0': x_next,
                    'rho': rho_next,
                    'error': error_list,
                    'energy': energy_list,
                }, os.path.join(args.save, '{:}_alph{:}_{:}_m{:}_n{}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m,n_tau)))
                print(f"breaking rel loss={rel_loss}")
                print(f"error saved in {os.path.join(args.save, '{:}_alph{:}_{:}_m{:}_n{}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m,n_tau))}")

                

            break


        if itr == 0 and n_tau == 0:
            with torch.no_grad():
                x0val,rhoval = toy_data.inf_train_gen(args.data, batch_size=1000)
                x0val  = cvt(torch.from_numpy(x0val))
                rhoval = cvt(torch.from_numpy(rhoval))
                plot_scatter_color(x0val, args, 0, rhoval, sPath = os.path.join(args.save, 'figs', f"rho_0.png"))

        previous_loss = loss

        log_message = (
            '{:05d}  {:6.3f}   {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e}'.format(
                itr, time_meter.val , loss, costs[0], costs[1], costs[2], rel_loss
            )
        )

        if loss.item() < global_loss:
            global_loss = loss.item()
            best_params = net.state_dict()


        if itr % 1000 == 0:
            net.load_state_dict(best_params)

        # create plots
        if itr % args.viz_freq == 0:
            print(log_message) # print iteration
            with torch.no_grad():
                # compute samples at the next time step
                x_next, rho_next = get_samples_next_time_step_including_det(x0val, rhoval, net, net_list, nt, n_tau)
                mean = torch.mean(x_next, dim=0)
                var  = torch.var(x_next)
                x_next, rho_next = get_samples_next_time_step_including_det(x0val, rhoval, net, net_list, nt, n_tau)
                plot_scatter_color(x_next, args, n_tau, rho_next, sPath = os.path.join(args.save, 'figs', f'n_tau_{n_tau}_itr_{itr}.png'))

        # shrink step size
        if itr % args.drop_freq == 0:
            for p in optim.param_groups:
                p['lr'] /= args.lr_drop
            # print("lr: ", p['lr'])

        end = time.time()

    print("Training Time: {:} seconds".format(time_meter.sum))
    print('Training has finished.  ' + '{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m))





