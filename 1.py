explanation = """
Keterangan Persiapan Data & Visualisasi Korelasi:
- Dataset yang digunakan adalah California Housing dari scikit-learn.
- Data dikonversi ke bentuk DataFrame dan kolom target 'Harga' dikalikan 100000 untuk satuan dollar.
- Lima data pertama ditampilkan untuk melihat struktur awal data.
- Statistik deskriptif menunjukkan nilai min, max, mean, dan quartiles untuk masing-masing fitur.
- Matriks korelasi dihitung untuk mengetahui hubungan antara setiap fitur dengan Harga rumah.
- Visualisasi heatmap membantu melihat korelasi antar variabel dengan warna (lebih terang = korelasi tinggi).
"""
print(explanation)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Harga'] = data.target * 100000

# Lihat data awal
print("\n Lima data pertama:")
print(df.head())

# Statistik deskriptif
print("\n Statistik deskriptif dataset:")
print(df.describe())

# Korelasi & visualisasi
print("\n Korelasi antara setiap fitur dan harga rumah:")
correlation_matrix = df.corr()
print(correlation_matrix['Harga'].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
plt.matshow(df.corr(), fignum=1)
plt.xticks(range(len(df.columns)), df.columns, rotation=90)
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar()
plt.title('Matriks Korelasi')
plt.show()
