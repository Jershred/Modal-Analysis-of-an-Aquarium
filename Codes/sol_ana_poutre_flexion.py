# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 01:52:30 2023

@author: Bastien Piron
"""
import numpy as np

import Poutre_flexion

def det(x):
    
    return 1-np.cos(x)*np.cosh(x)


def dichotomie(a,b):
    e=10e-12
    while((b-a)>e):
        m=(a+b)/2.0
        if(det(a)*det(m)<=0.0):
            b=m
        else:
            a=m
    
    return a


L=1.0;#m
b=0.01;#m
h=0.01;#m
e=10e-5
E=205.0e9;

N=4
dx=L/N
I=b*h*(b**2+h**2)/12.0
rho=7800.0;
S=b*h



W,V=Poutre_flexion.Analyse_modale(N, dx, L, rho, S, E, I, b, h)

for i in range((N-1)*2):
    
    arg=np.pi*0.5*(3+2*i)
    a=dichotomie(arg-0.25, arg+0.25)
    w=(a/L)**2*np.sqrt(E*I/(rho*S))/(2.0*np.pi)
    print('Fréquence exacte: ',w,'fréquence numérique: ', W[i], 'erreur relative: ', 100.0*abs(w-W[i])/w,'%')
    
    
    
    
    
    
    