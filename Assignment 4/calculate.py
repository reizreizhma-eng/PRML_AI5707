import math
import numpy as np 
import matplotlib.pyplot as plt 

seed =42
N=503*2

mu1=2
mu2=4
sigma=1

D1=[]
D2=[]
for i in range (N):
    seed = (seed * 1103515245 + 12345) % (2**31)
    u1 = seed / (2**31)
    seed = (seed * 1103515245 + 12345) % (2**31)
    u2 = seed / (2**31)
    if u1==0:
       u1=1e-10

    r = math.sqrt(-2 * math.log(u1))
        
    z1 = r * math.cos(2 * math.pi * u2)
    z2 = r * math.sin(2 * math.pi * u2)

    D1.append(mu1 + sigma * z1)
    D2.append(mu2 + sigma * z2)

D1=np.array(D1)
D2=np.array(D2)
X = np.column_stack((D1, D2))
mu = np.array([mu1, mu2])

print("First 10 values of D1:")
print(np.round(D1[:10],2))

print("First 10 values of D2:")
print(np.round(D2[:10],2))

