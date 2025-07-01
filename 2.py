explanation = """
Keterangan Regresi Linier Sederhana:
- Model ini menggunakan satu fitur yaitu 'MedInc' (Pendapatan Median) karena korelasinya paling tinggi terhadap 'Harga'.
- Data dibagi menjadi data latih dan data uji dengan perbandingan 80:20.
- Model Linear Regression dilatih menggunakan data latih.
- Dilakukan prediksi terhadap data uji untuk menghitung MSE dan R-squared.
- MSE menunjukkan seberapa jauh prediksi dari nilai aktual.
- R-squared menunjukkan seberapa besar variasi target dijelaskan oleh model.
- Visualisasi: titik biru adalah data aktual, garis merah adalah prediksi model.
"""
print(explanation)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = df[['MedInc']]
y = df['Harga']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f" Koefisien: {model.coef_[0]:.2f}")
print(f" Intercept: {model.intercept_:.2f}")
print(f" MSE: {mse:.2f}")
print(f" R-squared: {r2:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Data Aktual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Prediksi Model')
plt.xlabel('Pendapatan Median (puluhan ribu dollar)')
plt.ylabel('Harga Rumah (dollar)')
plt.title('Regresi Linear Sederhana: Pendapatan vs Harga Rumah')
plt.legend()
plt.grid(True)
plt.show()
