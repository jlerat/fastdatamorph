#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : Julien Lerat, CSIRO Environment
## Created : 2024-11-22 15:35:40.488686
## Comment : The morph algorithm aims at transforming a set of 2D
##           points while preserving means, stds and pearson correlation
##           as presented by Stefanie Mollin at PyCon 2024:
##           https://2024.pycon.org.au/program/3JHULA/
##
##           Stefanie Mollin algorithm relies on simulated annealing
##           optimisation technique. Here a variant of the algorithm is
##           proposed without using optimisation:
##
##           1. Both variables in the source dataset are rescaled so
##              that both means and stds match the target dataset.
##
##           2. A set of intermediate "states" are then generated as
##              linear combinations of source and target with weights
##              varying progressively from 0 (=source) to 1 (=target)
##
##           3. Pearson correlation is finally adjusted for each intermediate
##              state by realigning the set of points so that their
##              pearson correlation remains identical to the one of the target
##
##           The case study presented here maps 2 sets of points:
##           Source points : bi-variate gaussian distribution
##           Target points : Hypotrochoid parametric curve (star shape, see wikipedia)
##
## ------------------------------

import sys, os, re, json, math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

#----------------------------------------------------------------------
# @Config
#----------------------------------------------------------------------

# Number of points generated in both source and target datasets
npoints = 500

# Number of intermediate states between source and target
nsteps = 25

#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent

#----------------------------------------------------------------------
# @Utils
#----------------------------------------------------------------------
standard = lambda x: (x-x.mean())/x.std()

#----------------------------------------------------------------------
# @Get data
#----------------------------------------------------------------------

# 1. Create target data points
t = np.linspace(0, 1, npoints)*2*math.pi
fx = lambda t: 2*np.cos(4*t)+5*np.cos(8*t/3)
fy = lambda t: 2*np.sin(4*t)-5*np.sin(8*t/3)
xtarget = fx(t)
ytarget = fy(t)

# 2. Summary stats of target
rho = np.corrcoef(xtarget, ytarget)[0, 1]
xmean, xstd = xtarget.mean(), xtarget.std()
ymean, ystd = ytarget.mean(), ytarget.std()

# Create source points (random)
corr = 0.6
cov = xstd*ystd*corr
xy = mvn.rvs(mean=[xmean, ymean], cov=[[xstd**2, cov], [cov, ystd**2]], \
                size=npoints)
xsrc, ysrc = xy.T

# .. correct correlation
def correct_correlation(x, y, rho):
    xs = standard(x)
    ys = standard(y)

    c = np.sum(xs*ys)/npoints
    r = rho/c
    D = 4*r**4*(c**2-1)**2-4*(1-r**2)*(c**2-1)*r**2
    a = r**2*(c**2-1)/(1-r**2)+math.sqrt(D)/2/(1-r**2)
    ystar = y.mean()+y.std()*(a*c*xs+(1-a)*(ys-c*xs))
    return ystar

ysrc = correct_correlation(xsrc, ysrc, rho)

# .. standardise to get the same mean and std than target
xsrc = xmean+xstd*(xsrc-xsrc.mean())/xsrc.std()
ysrc = ymean+ystd*(ysrc-ysrc.mean())/ysrc.std()

#----------------------------------------------------------------------
# @Process
#----------------------------------------------------------------------

# Create intermediate sets
eps = (np.linspace(0, 1, nsteps+1))**0.2
xp = xsrc[:, None]*(1-eps)+xtarget[:, None]*eps
xp = xmean+xstd*(xp-xmean)/xp.std(axis=0)[None, :]

yp = ysrc[:, None]*(1-eps)+ytarget[:, None]*eps
yp = ymean+ystd*(yp-ymean)/yp.std(axis=0)[None, :]

# Correct correlation
check = np.zeros(nsteps)
for i in range(nsteps):
    xx, yy = xp[:, i], yp[:, i]
    yys = correct_correlation(xx, yy, rho)
    yp[:, i] = yys

#----------------------------------------------------------------------
# @Plot
#----------------------------------------------------------------------
plt.close("all")
fig, axs = plt.subplots(ncols=5, nrows=5, \
                figsize=(8, 8), \
                layout="constrained")

for iax, ax in enumerate(axs.flat):
    u, v = xp[:, 1+iax], yp[:, iax]
    ax.plot(u, v, "o")

    um, vm = u.mean(), v.mean()
    us, vs = u.std(), v.std()
    rho = np.corrcoef(u, v)[0, 1]
    txt = f"X: {um:0.2f} {us:0.2f}\n"\
            +f"Y: {vm:0.2f} {vs:0.2f}\n"\
            +f"R: {rho:0.2f}"

    ax.text(um, vm, txt, va="center", \
                ha="center", \
                fontsize=6, \
                path_effects=[pe.withStroke(linewidth=4, foreground="w")])
    ax.axis("off")

fp = froot / "results.png"
fig.savefig(fp, dpi=100)
