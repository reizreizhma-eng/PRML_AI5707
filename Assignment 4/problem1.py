import numpy as np
import matplotlib.pyplot as plt
from calculation import X

mean1 = np.mean(X, axis=0)
cov1 = np.cov(X, rowvar=False)
print("Estimated mean:")
print(mean1)
print("Estimated covariance:")
print(cov1)

eigenvalue1,eigenvector1 =np.linalg.eigh(cov1)
order=np.argsort(eigenvalue1)[::-1]
eigenvalue1=eigenvalue1[order]
eigenvector1=eigenvector1[:,order]

print("\nEigenvalues:")
print(eigenvalue1)

print("\nEigenvectors:")
print(eigenvector1)

x1 = np.linspace(mean1[0] - 4, mean1[0] + 4, 100)
x2 = np.linspace(mean1[1] - 4, mean1[1] + 4, 100)

X1_grid, X2_grid = np.meshgrid(x1, x2)

points = np.column_stack(
    (X1_grid.ravel(), X2_grid.ravel())
)

diff = points - mean1
cov_inv = np.linalg.inv(cov1)
density = np.sum((diff @ cov_inv) * diff,axis=1)
density = density.reshape(X1_grid.shape)

plt.figure()
plt.scatter(X[:, 0],X[:, 1],s=5)
plt.contour(X1_grid,X2_grid,density,levels=[1, 2, 3, 4, 5])
plt.scatter(mean1[0],mean1[1],marker="x",s=100)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Problem 1 - Constant Density Curves")
plt.grid()
plt.axis("equal")
plt.show()

major_vector1 = eigenvector1[:, 0]
minor_vector1 = eigenvector1[:, 1]
major_length1 = np.sqrt(eigenvalue1[0])
minor_length1 = np.sqrt(eigenvalue1[1])

plt.figure()
plt.scatter(X[:, 0],X[:, 1],s=5)
plt.scatter(mean1[0],mean1[1],marker="x",s=100)

plt.plot(
    [mean1[0] - major_vector1[0] * major_length1,mean1[0] + major_vector1[0] * major_length1],
    [mean1[1] - major_vector1[1] * major_length1,mean1[1] + major_vector1[1] * major_length1],color="red",label="Major Axis")


plt.plot(
    [mean1[0] - minor_vector1[0] * minor_length1,
    mean1[0] + minor_vector1[0] * minor_length1],
    [mean1[1] - minor_vector1[1] * minor_length1,
    mean1[1] + minor_vector1[1] * minor_length1],color="black",label="Minor Axis")

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Problem 1 - Major and Minor Axes")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()

print("\nMajor Eigenvalue:", eigenvalue1[0])
print("Major Eigenvector:", major_vector1)
print("Major Semi-Axis Length:", major_length1)
print("\nMinor Eigenvalue:", eigenvalue1[1])
print("Minor Eigenvector:", minor_vector1)
print("Minor Semi-Axis Length:", minor_length1)
