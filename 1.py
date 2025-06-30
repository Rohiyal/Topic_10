import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
Y = np.array([200, 400, 600, 800, 1000])

model = LinearRegression().fit(X, Y)
prediksi = model.predict([[35]])

print("Model: Y = {:.2f}X + {:.2f}".format(model.coef_[0], model.intercept_))
print("Prediksi pendapatan jika pelanggan 35 orang: {:.2f} ribu rupiah".format(prediksi[0]))
