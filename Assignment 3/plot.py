from polyregression import train_model, predict, mse, train, test, validation, data
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))
for degree in range(1, 11):
    W = train_model(train, degree)
    x_values = []
    y_values = []
    for row in data:
        x_values.append(row[0])
        y_values.append(row[1])

    predicted = []
    for x in x_values:
        y_pred = predict(x, W)
        predicted.append(y_pred)
    plt.subplot(2, 5, degree)
    plt.scatter(x_values,y_values,s=3)
    plt.plot(x_values,predicted,color="red")
    plt.title("Degree " + str(degree))
    plt.xlabel("x")
    plt.ylabel("y")

    plt.grid()


plt.tight_layout()


degrees = []

train_errors = []

test_errors = []

validation_errors = []


for degree in range(1, 11):

    W = train_model(
        train,
        degree
    )


    train_error = mse(
        train,
        W
    )


    test_error = mse(
        test,
        W
    )


    validation_error = mse(
        validation,
        W
    )


    degrees.append(
        degree
    )


    train_errors.append(
        train_error
    )


    test_errors.append(test_error)
    validation_errors.append(validation_error)
plt.figure(figsize=(8, 6))
plt.plot(degrees,train_errors,marker="o",label="Training data MSE")
plt.plot(degrees,test_errors,marker="o",label="Test data MSE")
plt.plot(degrees,validation_errors,marker="o",label="Validation data MSE")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("Effect of Polynomial Degree on MSE")
plt.xticks(degrees)
plt.legend()
plt.grid()
plt.show()
