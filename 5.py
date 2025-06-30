import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [50, 2],
    [60, 3],
    [70, 4],
    [80, 5],
    [90, 6]
])
Y = np.array([400, 480, 560, 640, 720])

model = LinearRegression().fit(X, Y)
prediksi = model.predict([[75, 4]])

print("Model: Y = {:.2f}X1 + {:.2f}X2 + {:.2f}".format(model.coef_[0], model.coef_[1], model.intercept_))
print("Prediksi penjualan jika pengunjung = 75 dan menu baru = 4: {:.2f} ribu rupiah".format(prediksi[0]))
