# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 15:40:43 2023

@author: Bastien Piron
"""
import numpy as np
import cavite_2D_carre


Lx=1.0;#m
Ly=0.5;#m

Nx=2;#points de discrétisation selon x
Ny=3;#points de discrétisation selon y

NE=Nx*Ny

dx=float(Lx/(Nx-1));#pas selon x
dy=float(Ly/(Ny-1));#pas selon y    

rho_c=1.2
c=340.0
 
Aff_min=0


W_tri=np.zeros((NE,2),dtype="float64");
W_tri2=np.zeros((NE,1),dtype="float64");    
V_reel=np.zeros((NE,NE),dtype="float64");
w=np.zeros((NE,1),dtype="float64");





#Calcul des fréquences exactes:    
for i in range(Nx):
    for j in range(Ny):
    
        w[i*Ny+j]=c*np.sqrt(((i+1)*np.pi/Lx)**2+((j+1)*np.pi/Ly)**2)/(2.0*np.pi)
        
 
#tri des fréquences:             
for i in range(NE):
    W_tri[i,0]=w[i]
    W_tri[i,1]=i
 
idx = np.argsort(W_tri[:,0])

for j in range(NE):
    W_tri2[j] =W_tri[idx[j],0]


#Comparaison des fréquences:
    
W,V=cavite_2D_carre.Analyse_modale(dx, dy, c, rho_c, Lx, Ly, Nx, Ny)   
 
for j in range(NE):
   print('Fréquence exacte: ',W_tri2[j],'fréquence numérique: ', W[j], 'erreur relative: ', 100.0*abs(W_tri2[j]-W[j])/W_tri2[j],'%')
  



      
