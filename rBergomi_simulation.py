#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:15:45 2024

Simulation of rBergomi model, see "Pricing under rough volatility"
We use the simulation package https://github.com/ryanmccrickerd/rough_bergomi

@author: lucapelizzari
"""
import numpy as np
from rBergomi import rBergomi



def SimulationofrBergomi(M, N, T, phi, rho, K, X0, H, xi, eta, r):
    """
    Simulate rBergomi price, variance, Brownian motions, and the conditional driver

        I_t = int_0^t sqrt(V_s) dW^1_s,

    together with its quadratic variation

        [I]_t = int_0^t V_s ds.

    Returns
    -------
    X : array, shape (M, N+1)
        Simulated price paths.
    V : array, shape (M, N+1)
        Variance paths.
    I : array, shape (M, N+1)
        Cumulative integrated volatility Brownian integral.
    QV_I : array, shape (M, N+1)
        Cumulative quadratic variation of I.
    dI : array, shape (M, N)
        Increments of I.
    dQV : array, shape (M, N)
        Increments of QV_I.
    dW1, dW2, dB : arrays
        Brownian increments from the rBergomi simulator.
    """
    tt = np.linspace(0.0, T, N + 1)
    dt = T / N

    rB = rBergomi(N, M, T, -0.5 + H)

    dW1 = rB.dW1()
    dW2 = rB.dW2()

    Y = rB.Y(dW1)
    V = rB.V(Y, xi, eta)

    dB = rB.dB(dW1, dW2, rho)

    X = rB.S(V, dB)
    X = X0 * X * np.exp(r * tt)

    # dW1 often has shape (M, N, 1), so squeeze the last dimension.
    dW1_flat = dW1[:, :, 0] if dW1.ndim == 3 else dW1

    dI = np.sqrt(V[:, :-1]) * dW1_flat
    dQV = V[:, :-1] * dt

    I = np.zeros((M, N + 1))
    I[:, 1:] = np.cumsum(dI, axis=1)

    QV_I = np.zeros((M, N + 1))
    QV_I[:, 1:] = np.cumsum(dQV, axis=1)

    return X, V, I, QV_I, dI, dQV, dW1, dW2, dB