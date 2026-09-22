import numpy as np
import matplotlib.pyplot as plt
from calculation import X, mu

sigma1=2
sigma2=0.5
C=np.array([[sigma1**2,0],[0,sigma2**2]])
c_sqrt = np.array([[sigma1, 0],[0, sigma2]])
Y=mu+(X-mu)@c_sqrt.T
mean=np.mean(Y,axis=0)
cov=np.cov(Y,rowvar=False)

print("Required Covariance:")
print(C)

print("\nEstimated Mean:")
print(mean)

print("\nEstimated Covariance:")
print(cov)

eigenvalue, eigenvector = np.linalg.eigh(cov)
order = np.argsort(eigenvalue)[::-1]
eigenvalue= eigenvalue[order]
eigenvector= eigenvector[:,order]
print("\nEigenvalues:")
print(eigenvalue)
print("\nEigenvectors:")
print(eigenvector)

x1=np.linspace(mean[0]-7,mean[0]+7,100)
x2=np.linspace(mean[1]-3,mean[1]+3,100)

X1_grid, X2_grid = np.meshgrid(x1, x2)

points = np.column_stack((X1_grid.ravel(), X2_grid.ravel()))

diff = points - mean
cov_inv = np.linalg.inv(cov)
density = np.sum((diff @ cov_inv) * diff,axis=1)

density = density.reshape(X1_grid.shape)
plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], s=5)
plt.contour(    X1_grid,X2_grid, density,levels=[1, 2, 3, 4, 5])

plt.scatter(mean[0],mean[1],marker="x",s=100)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Problem 2 - Diagonal Covariance")
plt.grid()
plt.axis("equal")
plt.show()


major_vector = eigenvector[:, 0]
minor_vector = eigenvector[:, 1]
major_length = np.sqrt(eigenvalue[0])
minor_length = np.sqrt(eigenvalue[1])

plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], s=5)
plt.scatter(mean[0],mean[1],marker="x",s=100)
plt.plot([mean[0] - major_vector[0] * major_length,mean[0] + major_vector[0] * major_length],[mean[1] - major_vector[1] * major_length,mean[1] + major_vector[1] * major_length],color="red",label="Major Axis")

plt.plot(
    [
        mean[0] - minor_vector[0] * minor_length,
        mean[0] + minor_vector[0] * minor_length
    ],
    [
        mean[1] - minor_vector[1] * minor_length,
        mean[1] + minor_vector[1] * minor_length
    ],color="black",
    label="Minor Axis"
)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Problem 2 - Major and Minor Axes")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()
print("\nMajor Eigenvalue:", eigenvalue[0])
print("Major Eigenvector:", major_vector)
print("\nMinor Eigenvalue:", eigenvalue[1])
print("Minor Eigenvector:", minor_vector)