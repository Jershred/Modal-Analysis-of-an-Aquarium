# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:36:08 2023

@author: Bastien Piron
"""
import numpy as np
import Poutre_traction_compression

rho = 7800.0
E=205.0e9 # densité air
S = 0.001
L = 1.0
n=5

W,V=Poutre_traction_compression.Analyse_modale(n,rho, E, S, L)



for i in range(len(W)):
    
    arg=np.pi*0.5*(1+2*i)/(2.0*np.pi)
    w=np.sqrt(E/rho)*arg/L
    print('Fréquence exacte: ',w,'fréquence numérique: ', W[i], 'erreur relative: ', 100.0*abs(w-W[i])/w,'%')