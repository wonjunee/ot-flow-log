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
    '--data', choices=['square', 'swissroll', '8gaussians','2gaussians', '1gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='2gaussians'
)
parser.add_argument('--alph'  , type=str, default='1.0,1.0,0.0')
parser.add_argument('--save'    , type=str, default='experiments/cnf/toy')

args = parser.parse_args()
# get precision type
prec = torch.float64

# get timestamp for saving models
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
def_batch    = 1024

if __name__ == '__main__':

    torch.set_default_dtype(prec)
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)
    d      = 2

    args.alph = [float(item) for item in args.alph.split(',')]

    net_list = []

    indices_list = []
    n=20 # number of points for the trajectories
    for i in range(n):
        indices_list.append(np.random.randint(def_batch))

    T = 50
    x0_list = np.zeros((T,n,2))

    for i in range(1,T+1):
        filename = os.path.join(args.save, '{:}_alph{:}_{:}_m{:}_n{}_checkpt.pth'.format(args.data,int(args.alph[1]),int(args.alph[2]),32,i))
        checkpt  = torch.load(filename, map_location=lambda storage, loc: storage)
        x0 = checkpt['x0']
        rho = checkpt['rho']
        x0_list[i-1] = x0.numpy()[indices_list]

    print(x0_list.shape)
    
    # let's plot

    fig,ax = plt.subplots(1,1,figsize=(6,6))

    for i in range(n):
        ax.plot(x0_list[:,i,0],x0_list[:,i,1], 'o-')
    plt.show()