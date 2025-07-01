explanation = """
Keterangan Regresi Linier Berganda:
- Menggunakan semua fitur dalam dataset kecuali kolom target 'Harga'.
- Data dinormalisasi menggunakan StandardScaler agar tiap fitur berada dalam skala yang sama.
- Model Linear Regression dilatih dengan data yang telah dinormalisasi.
- Dilakukan prediksi dan evaluasi terhadap data uji.
- MSE dan R-squared dihitung untuk menilai performa model.
- Koefisien setiap fitur ditampilkan untuk mengetahui seberapa besar kontribusinya terhadap prediksi harga rumah.
- Koefisien diurutkan untuk memudahkan interpretasi fitur yang paling berpengaruh.
"""
print(explanation)

from sklearn.preprocessing import StandardScaler

X = df.drop('Harga', axis=1)
y = df['Harga']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Hasil Regresi Linear Berganda:")
print(f"MSE: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

coefficients = pd.DataFrame({
    'Fitur': X.columns,
    'Koefisien': model.coef_
}).sort_values(by='Koefisien', ascending=False)

print("\n Koefisien Model:")
print(coefficients)
