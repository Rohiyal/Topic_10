explanation = """
Keterangan Regresi Polinomial (Degree 2):
- Digunakan untuk menangkap hubungan non-linear antara fitur dan target.
- Fitur yang digunakan tetap 'MedInc' (Pendapatan Median), tapi diubah ke bentuk polinomial orde 2.
- Pipeline digunakan untuk membuat proses fitur polinomial dan pelatihan model lebih sederhana.
- Model dilatih dan dilakukan prediksi terhadap data uji.
- Nilai MSE dan R-squared dihitung untuk menilai performa.
- Visualisasi: titik biru adalah data aktual, kurva merah adalah hasil prediksi model polinomial.
- Urutan plotting diurutkan terlebih dahulu agar kurva prediksi lebih halus.
"""
print(explanation)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X = df[['MedInc']]
y = df['Harga']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\nHasil Regresi Polinomial (degree = 2):")
print(f"MSE: {mse_poly:.2f}")
print(f"R-squared: {r2_poly:.2f}")

# Visualisasi
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Data Aktual')

sorted_idx = np.argsort(X_test.values.flatten())
plt.plot(X_test.values[sorted_idx], y_pred_poly[sorted_idx], color='red', linewidth=2, label='Prediksi Polinomial')

plt.xlabel('Pendapatan Median (puluhan ribu dollar)')
plt.ylabel('Harga Rumah (dollar)')
plt.title('Regresi Polinomial: Pendapatan vs Harga Rumah')
plt.legend()
plt.grid(True)
plt.show()
