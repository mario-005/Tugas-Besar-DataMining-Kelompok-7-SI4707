import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

@st.cache_data
def load_data():
    return pd.read_excel("pet_adoption_data.xlsx")

df = load_data()
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

st.sidebar.title("Data Analysis App")
menu = st.sidebar.radio("Pilih Menu", [
    "📂 Data Exploration", "🗑️ Data Preparation","📈 Data Understanding",  "📊 Correlation Heatmap",
    "🔍 Logistic Regression",
    "📌 Elbow Method", "📦 KMeans Clustering", "💡 Rekomendasi Adopsi"
])


# --- Menu: Data Exploration ---
if menu == "📂 Data Exploration":
    st.title("Data Exploration")

    st.subheader("1. Informasi Umum")
    st.write("Jumlah baris: ", len(df))
    st.write("Jumlah kolom: ", len(df.columns))
    st.dataframe(df)


    st.subheader("2. Statistik deskriptif ")
    st.write(df.describe())

    st.subheader("3. Informasi Tipe Data")
    st.write(df.dtypes)

    st.subheader("4. Data Unique")
    st.write(df.nunique())

# --- Menu: Data Preparation ---
elif menu == "🗑️ Data Preparation":
    st.title("**Data Preparation**")

    st.subheader("**1. Handling Missing Values**")
    df_clean = df.fillna(df.mean(numeric_only=True))
    st.write("Jumlah baris sebelum handling missing values: ", len(df))
    st.write("Jumlah baris setelah handling missing values: ", len(df_clean))

    st.subheader("**2. Data Cleaning**")
    df_clean = df.dropna()
    st.write("Jumlah baris sebelum data cleaning: ", len(df))
    st.write("Jumlah baris setelah data cleaning: ", len(df_clean))

    st.subheader("**3. Data Duplicate**")
    df_clean = df_clean.drop_duplicates()
    st.write("Jumlah baris sebelum data duplicate: ", len(df))
    st.write("Jumlah baris setelah data duplicate: ", len(df_clean))
    
    st.subheader("**4. Visualisasi Outlier**")
    numeric = df[['agemonths', 'weightkg']].dropna()
    fig, ax = plt.subplots()
    sns.boxplot(ax=ax, orient="v", data=numeric)
    st.pyplot(fig)
    st.markdown("""
Pada visualisasi outlier ini, kita menggunakan tabel `df` yang berisi data numerik. Tabel ini memiliki beberapa kolom yang digunakan dalam analisis, yaitu:

| Kolom | Deskripsi |
| --- | --- |
| `agemonths` | Umur dalam bulan |
| `WeightKg` | Berat badan dalam kg |

Tabel `df` ini merupakan sumber data utama yang digunakan dalam visualisasi outlier ini.
""")
    
    st.subheader("**5. Data Transformation**")
    numeric = df.select_dtypes(include=np.number).dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)
    df_clean = pd.DataFrame(scaled, columns=numeric.columns)
    st.write("Jumlah baris sebelum data transformation: ", len(df))
    st.write("Jumlah baris setelah data transformation: ", len(df_clean))



# --- Menu: Data Understanding (Pie/Bar Plots) ---
elif menu == "📈 Data Understanding":
    st.title("Distribusi Fitur Kategori")
    categorical_columns = df.select_dtypes(include='object').columns
    for col in categorical_columns:
        col1, col2, col3 = st.columns(3)
        with col1:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"Distribusi dari {col}")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title(f"Distribusi dari {col} (Pie)")
            st.pyplot(fig)
        with col3:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='line', ax=ax)
            ax.set_title(f"Distribusi dari {col} (Line)")
            st.pyplot(fig)
            
# --- Menu: Correlation Heatmap ---
elif menu == "📊 Correlation Heatmap":
    st.title("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)




# --- Menu: Logistic Regression (Target: Vaccinated) ---
elif menu == "🔍 Logistic Regression":
    st.title("Logistic Regression untuk Memprediksi Status Vaksinasi")

    df_model = df.dropna(subset=['vaccinated', 'pettype', 'breed', 'agemonths', 'color'])

    st.write("Contoh nilai unik 'vaccinated':", df_model['vaccinated'].unique())
    st.write("angka 1 sebagai true dan angka 0 sebagai false")

    if 'vaccinated' not in df_model.columns:
        st.warning("Kolom 'vaccinated' tidak ditemukan di data.")
    else:
        
        y = df_model['vaccinated'].astype(int)

        features = ['pettype', 'breed', 'agemonths', 'color']
        X = df_model[features].copy()

        X_encoded = pd.get_dummies(X, columns=['pettype', 'breed', 'color'], drop_first=True)


        if 'agemonths' in X_encoded.columns:
            scaler = StandardScaler()
            X_encoded[['agemonths']] = scaler.fit_transform(X_encoded[['agemonths']])

        model = LogisticRegression(max_iter=1000)
        model.fit(X_encoded, y)


        y_pred = model.predict(X_encoded)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.markdown("""
        **Confusion Matrix** menunjukkan jumlah prediksi yang **benar** dan **salah**:
        - **True Positive (TP)**: Diprediksi vaksinasi dan memang divaksin.
        - **True Negative (TN)**: Diprediksi tidak vaksinasi dan memang tidak divaksin.
        - **False Positive (FP)**: Diprediksi vaksinasi, tapi sebenarnya tidak.
        - **False Negative (FN)**: Diprediksi tidak vaksinasi, padahal sebenarnya sudah.
        """)

        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))

        st.markdown("""
        **Classification Report** memberikan metrik evaluasi:
        - **Precision**: Dari semua yang diprediksi sebagai 1, berapa yang benar?
        - **Recall**: Dari semua yang sebenarnya 1, berapa yang berhasil ditemukan?
        - **F1-score**: Rata-rata harmonik precision dan recall.
        - Cocok digunakan untuk melihat **keseimbangan performa model**.
        """)

        st.subheader("Accuracy")
        accuracy = accuracy_score(y, y_pred)
        st.write(f"{accuracy * 100:.2f}%")

        st.markdown(f"""
        **Accuracy** menunjukkan seberapa banyak prediksi yang benar dari seluruh data.
        Dalam kasus ini, model berhasil memprediksi dengan akurasi **{accuracy * 100:.2f}%**.
        """)

        st.subheader("**Diagram Feature Importance**")
        feature_importance = model.coef_[0]
        feature_names = X_encoded.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        feature_importance_df['Importance'] = feature_importance_df['Importance'].abs()  # mengambil nilai absolut
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
        ax.set_title('Feature Importance')
        st.pyplot(fig)


# --- Menu: Elbow Method ---
elif menu == "📌 Elbow Method":
    st.title("📌 Menentukan Jumlah Cluster Optimal (Elbow Method)")

    st.markdown("""
    **Apa itu Elbow Method?**  
    Elbow Method digunakan untuk menentukan jumlah _cluster_ optimal pada algoritma K-Means.  
    Kita mencari titik di mana penurunan _inertia_ (jumlah kuadrat jarak dalam cluster) mulai melambat — seperti "siku" pada grafik.

    **Inertia (Distortion):**  
    Merupakan total jarak kuadrat antara data dan pusat clusternya. Semakin kecil nilainya, semakin baik.  
    Namun, jika terlalu banyak cluster, penurunan inertia menjadi tidak signifikan (overfitting).
    """)

    numeric = df.select_dtypes(include=np.number).dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)

    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(scaled)
        distortions.append(km.inertia_)

    inertia_df = pd.DataFrame({
        "Jumlah Cluster (k)": range(1, 11),
        "Inertia": distortions
    })

    st.subheader("📊 Tabel Inertia per Jumlah Cluster")
    st.dataframe(inertia_df)


    st.subheader("📈 Grafik Elbow Curve")
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), distortions, marker='o')
    ax.set_xlabel('Jumlah Cluster (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method untuk Menentukan Cluster Optimal')
    st.pyplot(fig)

# --- Menu: KMeans Clustering ---
elif menu == "📦 KMeans Clustering":
    st.title("Hasil KMeans Clustering")

    numeric = df.select_dtypes(include=np.number).dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(pca_data)

    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
    ax.set_title("Cluster Visualisation (PCA)")

    legend_labels = [f"Cluster {i}" for i in np.unique(clusters)]
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
               for i in np.unique(clusters)]
    ax.legend(handles, legend_labels, title="Cluster")

    st.pyplot(fig)

    st.markdown("""
    **Catatan:** Warna pada visualisasi menunjukkan kelompok (cluster) hasil algoritma KMeans. 
    Colormap yang digunakan adalah viridis, yang memberikan gradasi warna dari ungu → biru → hijau → kuning.
    Masing-masing warna mewakili satu cluster berbeda.

    **Penjelasan Lebih Detail:**
    - Clustering dilakukan berdasarkan data numerik seperti berat, tinggi, umur, atau fitur lainnya dalam dataset.
    - Data distandarisasi terlebih dahulu menggunakan StandardScaler agar setiap fitur memiliki skala yang sama.
    - Kemudian dilakukan reduksi dimensi menggunakan PCA menjadi 2 dimensi agar bisa divisualisasikan.
    - Clustering dilakukan menggunakan KMeans dengan 3 cluster.
    - Setiap titik pada grafik merepresentasikan 1 hewan adopsi, dan warnanya menunjukkan ia masuk ke cluster mana.
    - Dengan clustering ini, hewan-hewan dikelompokkan berdasarkan kemiripan karakteristik numeriknya.
    - Ini bisa digunakan untuk memahami pola umum dari hewan-hewan dalam dataset, misalnya: Cluster 0 bisa jadi hewan kecil muda, Cluster 1 hewan dewasa besar, dst.
    """)
# --- Menu: Rekomendasi Adopsi ---
elif menu == "💡 Rekomendasi Adopsi":
    st.title("Rekomendasi Adopsi Hewan")

    st.markdown("Silakan isi preferensi Anda, dan kami akan memberikan rekomendasi adopsi berdasarkan data.")
    st.markdown("""
**Klasifikasi Umur**

* Dog:
	+ Muda: 0-12
	+ Dewasa: 13-84
	+ Tua: > 84
* Cat:
	+ Muda: 1-6
	+ Dewasa: 6-83
	+ Tua: > 84
* Rabbit:
	+ Muda: 0-12
	+ Dewasa: 13-84
	+ Tua: > 84
* Bird:
	+ Muda: 4-12
	+ Dewasa: 13-36
	+ Tua: > 36
""")

    df.columns = df.columns.str.strip().str.lower()

    age_preference = st.selectbox("Usia Hewan yang Diinginkan", options=["Muda", "Dewasa", "Tua"])
    home_type = st.selectbox("Jenis Tempat Tinggal Anda", options=["Apartemen", "Rumah", "Lainnya"])
    pet_type = st.selectbox("Jenis Hewan Yang Diinginkan", options=["Dog", "Cat", "Rabbit", "Bird"])

    if st.button("Dapatkan Rekomendasi"):
        filtered = df.copy()
        if "agemonths" in df.columns:
            if pet_type.lower() == "bird":
                if age_preference == "Muda":
                    filtered = filtered[(filtered["agemonths"] >= 4) & (filtered["agemonths"] <= 12)]
                elif age_preference == "Dewasa":
                    filtered = filtered[(filtered["agemonths"] >= 13) & (filtered["agemonths"] <= 36)]
                elif age_preference == "Tua":
                    filtered = filtered[filtered["agemonths"] > 36]

            elif pet_type.lower() == "Cat":
                if age_preference == "Muda":
                    filtered = filtered[(filtered["agemonths"] >= 1) & (filtered["agemonths"] <= 6)]
                elif age_preference == "Dewasa":
                    filtered = filtered[(filtered["agemonths"] >= 6) & (filtered["agemonths"] <= 83)]
                elif age_preference == "Tua":
                    filtered = filtered[filtered["agemonths"] > 84]

            elif pet_type.lower() == "Bird":
                if age_preference == "Muda":
                    filtered = filtered[(filtered["agemonths"] >= 0) & (filtered["agemonths"] <= 6)]
                elif age_preference == "Dewasa":
                    filtered = filtered[(filtered["agemonths"] >= 6) & (filtered["agemonths"] <= 60)]
                elif age_preference == "Tua":
                    filtered = filtered[filtered["agemonths"] > 61]
            else:
                if age_preference == "Muda":
                    filtered = filtered[filtered["agemonths"] <= 12]
                elif age_preference == "Dewasa":
                    filtered = filtered[(filtered["agemonths"] > 12) & (filtered["agemonths"] <= 84)]
                elif age_preference == "Tua":
                    filtered = filtered[filtered["agemonths"] > 84]

        if "pettype" in df.columns:
            filtered = filtered[filtered["pettype"].str.lower() == pet_type.lower()] 

        if not filtered.empty:
            st.success(f"Kami menemukan {len(filtered)} hewan yang sesuai dengan preferensi Anda:")
            show_cols = [col for col in ["pettype", "breed", "agemonths", "color"] if col in filtered.columns]
            st.dataframe(filtered[show_cols].head(30))
        else:
            st.warning("Maaf, tidak ada hewan yang cocok ditemukan berdasarkan preferensi tersebut.")
