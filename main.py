import numpy as np
import matplotlib.pyplot as plt


new = np.loadtxt('C:/Users/21055/Downloads/fuel consumption of car fleet.txt', skiprows=1)
print(new)
new_0 = new[:, 0]
new_3 = new[:, 3]


def predict(x, w, b):  # Calculation for all X in the array
    return x * w + b


def loss(x, y, w, b):
    return np.average((predict(x, w, b) - y) ** 2)


def train(x, y, il, lr):
    # lr is defined as the learning rate or step size with which the parameters w and b are modified
    w = b = 0  # starting values --> can be important to reach a good fit

    for i in range(il):
        current_loss = loss(x, y, w, b)
        # print("Iteration %4d => Loss: %6f" % (i, current_loss))
        if loss(x, y, w + lr, b) < current_loss:
            w += lr
        elif loss(x, y, w - lr, b) < current_loss:
            w -= lr
        elif loss(x, y, w, b + lr) < current_loss:
            b += lr
        elif loss(x, y, w, b - lr) < current_loss:
            b -= lr
        else:
            return w, b

    # raise Exception("Could not coverage within %d iterations." % (iter))  # raising your own defined error


w0, b0 = train(new_0, new_3, il=10000, lr=0.01)
print("\nw=%.3f, b=%.3f" % (w0, b0))


plt.plot(new_0, new_3, 'bo')
plt.plot(new_0, new_0*w0+b0)
plt.xlabel('Fuel Consumption/L')
plt.ylabel('Distance/(km/10)')
plt.show()
