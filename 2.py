import numpy as np

X = np.array([1, 2, 3, 4])
Y = np.array([60, 65, 70, 75])

peningkatan_per_jam = (Y[-1] - Y[0]) / (X[-1] - X[0])
print("Rata-rata peningkatan nilai per jam belajar: {:.2f} poin".format(peningkatan_per_jam))
