data = []

with open("noisy_11.txt", "r") as file:
    for line in file:
        x,y = line.split()
        data.append([float(x), float(y)])

print("Total data:",len(data))

#split data
train= data[0::5]+data[2::5]+data[4::5]
test= data[1::5]
validation= data[3::5]

print("Train data     :", len(train))
print("Test data      :", len(test))
print("Validation data:", len(validation))
# TRANSPOSE
def transpose(A):
    T = []
    for j in range(len(A[0])):
        row = []
        for i in range(len(A)):
            row.append(A[i][j])
        T.append(row)
    return T

#matrix multy
def multiply(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            value = 0
            for k in range(len(B)):
                value = value + A[i][k] * B[k][j]
            row.append(value)
        result.append(row)
    return result

#determent
def determinant(A):
    if len(A) == 1:
        return A[0][0]
    if len(A) == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    det = 0
    for j in range(len(A)):
        minor = []
        for i in range(1, len(A)):
            row = []
            for k in range(len(A)):
                if k != j:
                    row.append(A[i][k])
            minor.append(row)
        det = det + ((-1) ** j) * A[0][j] * determinant(minor)
    return det
#inverse
def inverse(A):
    det = determinant(A)
    if det == 0:
        raise ValueError("Inverse does not exist")
    n = len(A)
    cofactor = []
    for i in range(n):
        row = []
        for j in range(n):
            minor = []
            for r in range(n):
                if r == i:
                    continue
                minor_row = []
                for c in range(n):
                    if c != j:
                        minor_row.append(A[r][c])
                minor.append(minor_row)
            value = ((-1) ** (i + j)) * determinant(minor)
            row.append(value)
        cofactor.append(row)

    adjoint = transpose(cofactor)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(adjoint[i][j] / det)
        result.append(row)

    return result


#creat x
def create_X(data, degree):
    X = []
    for x, y in data:
        row = []
        for power in range(degree + 1):
            row.append(x ** power)
        X.append(row)
    return X
#creat Y
def create_Y(data):
    Y = []
    for x, y in data:
        Y.append([y])
    return Y
#Train 
def train_model(data, degree):
    X = create_X(data, degree)
    Y = create_Y(data)
    XT = transpose(X)
    XTX = multiply(XT, X)
    XTY = multiply(XT, Y)
    # Weight
    XTX_inverse = inverse(XTX)
    W_matrix = multiply(XTX_inverse, XTY)
    W = []
    for i in range(len(W_matrix)):
        W.append(W_matrix[i][0])
    return W

#predect
def predict(x, W):
    y = 0
    for i in range(len(W)):
        y = y + W[i] * (x ** i)
    return y
#mse
def mse(data, W):
    error = 0
    for x, y in data:
        predicted = predict(x, W)
        error = error + (y - predicted) ** 2
    return error / len(data)
#best degree
best_degree = 0
best_test_error = None
best_W = None
print("\nDegree\tTrain MSE\tTest MSE")
print("--------------------------------")
for degree in range(1, 11):
    W = train_model(train, degree)
    train_error = mse(train, W)
    test_error = mse(test, W)
    print(
        degree,
        "\t",
        round(train_error, 4),
        "\t",
        round(test_error, 4)
    )
    if best_test_error is None or test_error < best_test_error:
        best_test_error = test_error
        best_degree = degree

        best_W = W

#validation
validation_error = mse(validation, best_W)

#result
print("\n----------------------------")
print("Best Degree    :", best_degree)
print("Test MSE       :", round(best_test_error, 4))
print("Validation MSE :", round(validation_error, 4))
print("----------------------------")
#weight
print("\nWeights:")
for i in range(len(best_W)):
    print(
        "w" + str(i),
        "=",
        best_W[i]
    )