# 🐾 Aplikasi Analisis dan Rekomendasi Adopsi Hewan

Aplikasi berbasis **Streamlit** untuk eksplorasi, analisis, prediksi, dan rekomendasi adopsi hewan berdasarkan dataset adopsi. Aplikasi ini ditujukan untuk membantu organisasi atau individu dalam memahami data hewan adopsi serta memberikan saran berbasis data.

---

## 📚 Fitur Utama

| Menu | Deskripsi |
|------|-----------|
| 📂 **Data Exploration** | Menampilkan ringkasan data, statistik deskriptif, tipe data, dan jumlah nilai unik. |
| 🗑️ **Data Preparation** | Menangani missing values, cleaning, duplikasi, outlier, dan transformasi data numerik. |
| 📈 **Data Understanding** | Visualisasi distribusi fitur kategori dalam bentuk bar, pie, dan line chart. |
| 📊 **Correlation Heatmap** | Menampilkan korelasi antar fitur numerik. |
| 🔍 **Logistic Regression** | Memprediksi status vaksinasi hewan berdasarkan fitur-fitur seperti jenis hewan, umur, warna, dll. |
| 📌 **Elbow Method** | Menentukan jumlah cluster optimal untuk KMeans Clustering. |
| 📦 **KMeans Clustering** | Visualisasi hasil clustering hewan menggunakan PCA. |
| 💡 **Rekomendasi Adopsi** | Memberikan saran adopsi hewan berdasarkan preferensi pengguna (jenis hewan, umur, dll). |

---

## 📁 Struktur File

- `app.py` – File utama aplikasi Streamlit.
- `pet_adoption_data.xlsx` – Dataset adopsi hewan yang digunakan.

> **Catatan**: Pastikan file `pet_adoption_data.xlsx` tersedia di direktori yang sama saat menjalankan aplikasi.

---

## ▶️ Cara Menjalankan

1. Pastikan Anda memiliki Python 3.x dan pip.
2. Instal semua dependensi berikut:
3. Streamlit run app.py
```bash
pip install streamlit pandas matplotlib seaborn scikit-learn openpyxl
