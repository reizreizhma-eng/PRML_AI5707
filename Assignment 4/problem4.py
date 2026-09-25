import numpy as np
import matplotlib.pyplot as plt
from calculation import X,mu

Z=X-mu

mu_class1=np.array([2,5])
mu_class2=np.array([7,5])

# Case 1: C1=C2=sigma^2 I

C1=np.array([[1,0],[0,1]])
C2=np.array([[1,0],[0,1]])

Y1=mu_class1+Z
Y2=mu_class2+Z

C_inv=np.linalg.inv(C1)
w=C_inv@(mu_class1-mu_class2)
w0=-0.5*(mu_class1.T@C_inv@mu_class1-mu_class2.T@C_inv@mu_class2)

x=np.linspace(-3,12,200)

if abs(w[1])>1e-10:
    y=-(w[0]*x+w0)/w[1]
else:
    y=None
    x_line=-w0/w[0]

plt.figure()
plt.scatter(Y1[:,0],Y1[:,1],s=5,label="Class 1")
plt.scatter(Y2[:,0],Y2[:,1],s=5,label="Class 2")

if y is not None:
    plt.plot(x,y,label="Decision Boundary")
else:
    plt.axvline(x=x_line,label="Decision Boundary")

plt.scatter(mu_class1[0],mu_class1[1],marker="x",s=100)
plt.scatter(mu_class2[0],mu_class2[1],marker="x",s=100)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Case 1: C1=C2=sigma^2 I")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()

print("\nCase 1")
print("C1:")
print(C1)
print("C2:")
print(C2)
print("Estimated Covariance Class 1:")
print(np.cov(Y1,rowvar=False))
print("Estimated Covariance Class 2:")
print(np.cov(Y2,rowvar=False))
print("Decision Boundary: Linear")


# Case 2: C1=C2=C

C1=np.array([[4,1],[1,1]])
C2=np.array([[4,1],[1,1]])

L=np.linalg.cholesky(C1)

Y1=mu_class1+Z@L.T
Y2=mu_class2+Z@L.T

C_inv=np.linalg.inv(C1)
w=C_inv@(mu_class1-mu_class2)
w0=-0.5*(mu_class1.T@C_inv@mu_class1-mu_class2.T@C_inv@mu_class2)

x=np.linspace(-5,14,200)

if abs(w[1])>1e-10:
    y=-(w[0]*x+w0)/w[1]
else:
    y=None
    x_line=-w0/w[0]

plt.figure()
plt.scatter(Y1[:,0],Y1[:,1],s=5,label="Class 1")
plt.scatter(Y2[:,0],Y2[:,1],s=5,label="Class 2")

if y is not None:
    plt.plot(x,y,label="Decision Boundary")
else:
    plt.axvline(x=x_line,label="Decision Boundary")

plt.scatter(mu_class1[0],mu_class1[1],marker="x",s=100)
plt.scatter(mu_class2[0],mu_class2[1],marker="x",s=100)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Case 2: C1=C2=C")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()

print("\nCase 2")
print("Common Covariance:")
print(C1)
print("Estimated Covariance Class 1:")
print(np.cov(Y1,rowvar=False))
print("Estimated Covariance Class 2:")
print(np.cov(Y2,rowvar=False))
print("Decision Boundary: Linear")


# Case 3: C1!=C2

C1=np.array([[4,1],[1,1]])
C2=np.array([[1,0],[0,4]])

L1=np.linalg.cholesky(C1)
L2=np.linalg.cholesky(C2)

Y1=mu_class1+Z@L1.T
Y2=mu_class2+Z@L2.T

inv_C1=np.linalg.inv(C1)
inv_C2=np.linalg.inv(C2)

det_C1=np.linalg.det(C1)
det_C2=np.linalg.det(C2)

x1=np.linspace(-5,14,200)
x2=np.linspace(-4,14,200)

X1_grid,X2_grid=np.meshgrid(x1,x2)

points=np.column_stack((X1_grid.ravel(),X2_grid.ravel()))

diff1=points-mu_class1
distance1=np.sum((diff1@inv_C1)*diff1,axis=1)

diff2=points-mu_class2
distance2=np.sum((diff2@inv_C2)*diff2,axis=1)

g1=-0.5*np.log(det_C1)-0.5*distance1
g2=-0.5*np.log(det_C2)-0.5*distance2

boundary=(g1-g2).reshape(X1_grid.shape)

plt.figure()
plt.scatter(Y1[:,0],Y1[:,1],s=5,label="Class 1")
plt.scatter(Y2[:,0],Y2[:,1],s=5,label="Class 2")
plt.contour(X1_grid,X2_grid,boundary,levels=[0])
plt.scatter(mu_class1[0],mu_class1[1],marker="x",s=100)
plt.scatter(mu_class2[0],mu_class2[1],marker="x",s=100)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Case 3: C1!=C2")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()

print("\nCase 3")
print("Class 1 Covariance:")
print(C1)
print("Class 2 Covariance:")
print(C2)
print("Estimated Covariance Class 1:")
print(np.cov(Y1,rowvar=False))
print("Estimated Covariance Class 2:")
print(np.cov(Y2,rowvar=False))
print("Decision Boundary: Quadratic")


print("\nConclusion")
print("Case 1: C1=C2=sigma^2 I -> Linear boundary")
print("Case 2: C1=C2=C -> Linear boundary")
print("Case 3: C1!=C2 -> Quadratic boundary")
