# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 14:06:33 2023

@author: Bastien Piron
"""

import numpy as np
import cavite_1D

n = 5
rho = 1.2 # densité air
S = 0.001
L = 5.0
c = 340.0

W,V=cavite_1D.Analyse_modale(n, rho, c, L, S)

if n<6:
    coeff=[0.0,9.87,39.47, 88.82, 157.1, 246.75]
    
    for i in range(len(W)):
        
        w=np.sqrt(coeff[i])*(c/L)/(2.0*np.pi)
        print('Fréquence exacte: ',w,'fréquence numérique: ', W[i], 'erreur relative: ', 100.0*abs(w-W[i])/w,'%')
        
else:
    print("n trop grand")