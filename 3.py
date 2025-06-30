import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([1, 3, 5, 7, 9]).reshape(-1, 1)
Y = np.array([150, 120, 90, 60, 30])

model = LinearRegression().fit(X, Y)
prediksi = model.predict([[4]])

print("Model: Y = {:.2f}X + {:.2f}".format(model.coef_[0], model.intercept_))
print("Prediksi harga jual jika usia kendaraan 4 tahun: {:.2f} juta rupiah".format(prediksi[0]))
