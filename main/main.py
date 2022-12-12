# -*- coding: utf-8 -*-
"""
@author: Sumit Patel

"""
import time
import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import interpolate
import matplotlib.pyplot as plt


def BS_Analytical(spot, strike, risk_free_rate, time_to_mat, vol, opt_type='call'):
    if opt_type != 'call' and opt_type != 'put':
        print("[Error] Wrong option type!!!")
        return 0;

    d1 = (np.log(spot / strike) + (risk_free_rate + vol ** 2 / 2) * time_to_mat) / (vol * np.sqrt(time_to_mat))
    d2 = d1 - vol * np.sqrt(time_to_mat)
    Nd1 = norm.cdf(d1, 0, 1)
    Nd2 = norm.cdf(d2, 0, 1)

    opt_price = Nd1 * spot - Nd2 * strike * np.exp(-risk_free_rate * time_to_mat)

    return (opt_price)


class BSPDE:
    def __init__(self, sMin, sMax, ns, tMin, tMax, nt, strike, risk_free_rate, vol, opt_type='EuropenaCall'):
        self.sMin = sMin
        self.sMax = sMax
        self.ns = ns
        self.tMin = tMin
        self.tMax = tMax
        self.nt = nt

        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.vol = vol
        self.opt_type = opt_type

        self.constructGrid()
        self.solvePDE()

    def constructGrid(self):
        s = np.linspace(self.sMin, self.sMax, self.ns)
        t = np.linspace(self.tMin, self.tMax, self.nt)
        self.tGrid, self.sGrid = np.meshgrid(t, s)

    def solvePDE(self):
        # Terminal condition on time
        self.pricePDE = np.zeros(self.tGrid.shape)
        self.pricePDE[:, 0] = [max(s - self.strike, 0) for s in self.sGrid[:, 0]]
        # Boundary condition on stock price
        self.pricePDE[0, :] = 0
        self.pricePDE[-1, :] = self.sGrid[-1, :] - self.strike * np.exp(-self.risk_free_rate * self.tGrid[-1, :])

        # Interior using FTCS
        for i in range(1, self.nt):
            dt = self.tGrid[0, i] - self.tGrid[0, i - 1]
            for j in range(1, self.ns - 1):
                ds = (self.sGrid[j + 1, i - 1] - self.sGrid[j - 1, i - 1]) / 2.

                dvds = (self.pricePDE[j + 1, i - 1] - self.pricePDE[j - 1, i - 1]) / (2. * ds)

                #                 dvds = (self.pricePDE[j,i-1] - self.pricePDE[j-1,i-1]) / (ds)

                d2vds2 = (self.pricePDE[j + 1, i - 1] - 2. * self.pricePDE[j, i - 1] + self.pricePDE[j - 1, i - 1]) / (
                            ds ** 2)
                rhs = self.risk_free_rate * self.pricePDE[j, i - 1] \
                      - self.risk_free_rate * self.sGrid[j, i - 1] * dvds \
                      - 0.5 * self.vol ** 2 * self.sGrid[j, i - 1] ** 2 * d2vds2

                self.pricePDE[j, i] = self.pricePDE[j, i - 1] - dt * rhs

        # Interpolation function of the PDE solution
        self.PDEPriceInterpolater = interpolate.interp2d(self.tGrid[0, :], self.sGrid[:, 0], self.pricePDE,
                                                         kind='linear')

    def getPDEprice(self, spot, time):
        p = self.PDEPriceInterpolater(time, spot)
        return (p)




if __name__ == '__main__':

    BS = BSPDE(80, 120, 41, 0, 1, 10001, 100, 0.02, 0.1, 'EuropenaCall')

    fig, ax = plt.subplots(1, 1)

    time_to_mats = np.linspace(1e-8, .25, 5)
    spots = np.linspace(95, 105, 11)

    for t in time_to_mats:
        call_prices_analytical = [BS_Analytical(x, 100, 0.02, t, 0.1) for x in spots]
        call_prices_PDE = [BS.getPDEprice(x, t) for x in spots]
        ax.plot(spots, call_prices_analytical,'-',label='Analytical')
        ax.plot(spots, call_prices_PDE,'.',label='PDE')
        ax.legend(['Analytical','PDE (FTCS)'])
        ax.set_xlabel('Spot', fontweight ='bold')
        ax.set_ylabel('Option Value', fontweight ='bold')
        ax.set_title('European Call, PDE', fontsize = 14, fontweight ='bold')
        ax.grid(True)
    
    fig.savefig('../Results/Plots/EuropeanCallPDE.pdf')
    plt.show()
    
    print("Completed")
