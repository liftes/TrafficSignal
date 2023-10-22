import model.global_Value as G
import model.traffic as gv
import model.energy as energy
import tools.file as ft

import math
import copy
import random
import numpy as np

def ReassignSigmaValues(sigma):
    """
    If sigma values exceed 1 or go below -1, 
    reassign them to random values between 0.1 and 0.3 or -0.3 and -0.1 respectively.
    """
    condition = np.abs(sigma) >= 1
    sigma[condition & (sigma >= 1)] = np.random.uniform(0.1, 0.3)
    sigma[condition & (sigma <= -1)] = np.random.uniform(-0.3, -0.1)
    return sigma

def UpdateSigma(bi_sigma, bi_sigma_last, bi_sigmat, t, L, a0, at):
    """
    Update bi_sigma values based on certain conditions and derivatives.
    """
    y = np.random.random(L)
    x_d = np.zeros(L)

    for i in range(L):
        if abs(bi_sigma[i]) < 1:
            y[i] -= ((a0 - at) * bi_sigma[i] + (HqDer(i, 1, bi_sigmat) + HdDer(bi_sigma[i], bi_sigma_last[i]) + HwDer(bi_sigmat[i], 1))) * t
            x_d[i] = a0 * y[i]
            bi_sigma[i] += x_d[i] * t
            bi_sigma[i] = np.clip(bi_sigma[i], -1, 1)
        else:
            bi_sigma[i] = np.clip(bi_sigma[i], -1, 1)
    return bi_sigma

def BsbRefactored(bi_sigma, bi_sigma_last):
    global bi_Sigma, bi_Sigma_last

    at = 0
    sigma_lists = [bi_sigma[:, i] for i in range(4)]
    sigma_last_lists = [bi_sigma_last[:, i] for i in range(4)]
    
    # Reassign sigma values that are out of range
    sigma_lists = [ReassignSigmaValues(sigma) for sigma in sigma_lists]
    bi_sigmat = Get4Bisigma(*sigma_lists)

    Hlist.append(HTotal(Param1(dict1, dict2, dict3, bi_sigmat), Param2(bi_sigma_last, bi_sigmat)))

    for _ in range(iters):
        for sigma, sigma_last in zip(sigma_lists, sigma_last_lists):
            sigma = UpdateSigma(sigma, sigma_last, bi_sigmat, t, L, a0, at)
            bi_sigmat = Get4Bisigma(*sigma_lists)

        at += delta

        Hamilton = HTotal(Param1(dict1, dict2, dict3, bi_sigmat), Param2(bi_sigma_last, bi_sigmat))
        Hlist.append(Hamilton)

    bi_sigmat = Normalize(bi_sigmat)
    bi_sigmat, sigma8 = GetSigma8(bi_sigmat)
    SigmaList.append(bi_sigmat)
    sigma8List.append(sigma8)

    HList.append(HTotal(Param1(dict1, dict2, dict3, bi_sigmat), Param2(bi_sigma_last, bi_sigmat)))

    bi_Sigma_last = bi_sigmat
    bi_Sigma = bi_sigmat



