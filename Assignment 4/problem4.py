import numpy as np
import matplotlib.pyplot as plt
from calculation import X, mu, N
Z = X-mu

mu_class1 = np.array([2, 5])
mu_class2 = np.array([7, 5])


C_equal = np.array([ [1, 0],[0, 1]])
Y1 = mu_class1 + Z
Y2 = mu_class2 + Z

C_inv = np.linalg.inv(C_equal)


w = C_inv @ (mu_class1 - mu_class2)
w0 = (-0.5* (mu_class1.T @ C_inv @ mu_class1- mu_class2.T @ C_inv @ mu_class2))

x_boundary = np.linspace(-3, 12, 100)
y_boundary = -(w[0] * x_boundary + w0) / w[1] if abs(w[1]) > 1e-10 else None

plt.figure()

plt.scatter(Y1[:, 0], Y1[:, 1],s=5,label="Class 1")
plt.scatter(Y2[:, 0],Y2[:, 1],s=5,label="Class 2")
if y_boundary is not None:

        plt.plot(x_boundary,y_boundary,label="Decision Boundary"    )

else:
        x_line = -w0 / w[0]

plt.axvline(x=x_line,label="Decision Boundary")
plt.scatter(mu_class1[0],mu_class1[1],marker="x",s=100)
plt.scatter(mu_class2[0],mu_class2[1],marker="x",s=100)

plt.xlabel("X1")
plt.ylabel("X2")

plt.title("Problem 4 - Equal Covariance Linear Boundary")

plt.legend()
plt.grid()
plt.axis("equal")

plt.show()
C_class1 = np.array([[4, 1], [1, 1]])
C_class2 = np.array([[1, 0], [0, 4]])

L1 = np.linalg.cholesky(C_class1)
L2 = np.linalg.cholesky(C_class2)
Y1 = mu_class1 + Z @ L1.T
Y2 = mu_class2 + Z @ L2.T
print("\nClass 1 Mean:")
print(mu_class1)

print("\nClass 1 Covariance:")
print(C_class1)

print("\nClass 2 Mean:")
print(mu_class2)

print("\nClass 2 Covariance:")
print(C_class2)

plt.figure()

plt.scatter(Y1[:, 0],Y1[:, 1],s=5,label="Class 1")
plt.scatter(Y2[:, 0],Y2[:, 1],s=5,label="Class 2")

plt.scatter(mu_class1[0],mu_class1[1],marker="x",s=100)
plt.scatter(mu_class2[0],mu_class2[1],marker="x", s=100)
plt.xlabel("X1")
plt.ylabel("X2")

plt.title( "Problem 4 - Different Covariance Classes")

plt.legend()
plt.grid()
plt.axis("equal")

plt.show()

inv_C1 = np.linalg.inv(C_class1)
inv_C2 = np.linalg.inv(C_class2)

det_C1 = np.linalg.det(C_class1)
det_C2 = np.linalg.det(C_class2)

x1 = np.linspace(-5, 14, 200)
x2 = np.linspace(-4, 14, 200)
X1_grid, X2_grid = np.meshgrid(x1, x2)
points = np.column_stack((X1_grid.ravel(),X2_grid.ravel()))

diff1 = points - mu_class1

distance1 = np.sum( (diff1 @ inv_C1) * diff1, axis=1)

g1 = (-0.5 * np.log(det_C1)-0.5 * distance1)
diff2 = points - mu_class2

distance2 = np.sum((diff2 @ inv_C2) * diff2, axis=1)

g2 = ( -0.5 * np.log(det_C2)   -0.5 * distance2)

boundary = g1 - g2
boundary = boundary.reshape(X1_grid.shape)
plt.figure()
plt.scatter(Y1[:, 0],Y1[:, 1], s=5,label="Class 1")


plt.scatter(Y2[:, 0],Y2[:, 1],s=5,label="Class 2")


plt.contour(X1_grid,X2_grid,boundary,levels=[0])


plt.scatter(mu_class1[0],mu_class1[1], marker="x",s=100)


plt.scatter(mu_class2[0], mu_class2[1],marker="x",s=100)

plt.xlabel("X1")
plt.ylabel("X2")

plt.title(
    "Problem 4 - Different Covariance Quadratic Boundary"
)

plt.legend()
plt.grid()
plt.axis("equal")

plt.show()
print("\nEstimated Covariance of Class 1:")
print(np.cov(Y1, rowvar=False))
print("\nEstimated Covariance of Class 2:")
print(np.cov(Y2, rowvar=False))
print("\nConclusion:")
print("Equal covariance matrices give a linear decision boundary.")