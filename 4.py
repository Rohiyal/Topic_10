import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([5, 10, 15, 20]).reshape(-1, 1)
Y = np.array([25000, 50000, 75000, 100000])

model = LinearRegression().fit(X, Y)
prediksi = model.predict([[18]])

print("Model: Y = {:.2f}X + {:.2f}".format(model.coef_[0], model.intercept_))
print("Prediksi biaya pulsa untuk 18 jam internet: Rp {:.0f}".format(prediksi[0]))
