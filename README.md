# **Tahapan Pembuatan Model Naive Bayes untuk Klasifikasi Kelayakan Kredit Computer**

1. **Melakukan loading library python yang akan digunakan.**
   - Model kali ini menggunakan library pandas, numpy, sklearn, matplotlib, seaborn, dan joblib.
2. **Mengambil dataset dan membuat dataframe**
   - Di tahap ini, data yang berasal dari csv akan dimasukan ke dalam dataset df_net dan dilakukan pemanggilan head dataset untuk mengonfirmasi apakah dataset sudah terjalankan atau belum.
3. **Melakukan Assessing Data**
   - Kita akan meninjau informasi awal data untuk melihat apakah ada yang perlu dilakukan cleaning data atau tidak.
4. **Identifikasi tipe Kolom**
   - Sebelum membuat model, dilakukan identifikasi tipe data kolom apakah terasuk kategorikal atau numerik untuk identifikasi awal sebuah model.
5. **Pengecekan missing value**
   - Pada bagian ini akan dicek apakah ada missing value yang perlu ditangani atau tidak.
6. **Meninjau distribusi predictor terhadap target**
   - Sebelum membuat model, terlebih dahulu dilakukan visualisasi distribusi predictor terhadap target guna meninjau gambaran besar dataset.
7. **Memisahkan predictor dari target**
   - Pada tahap ini,Buys_Computer yang merupakan target dihapus dari predictor (**x**) dan dimasukan ke dalam variable target (**y**).
8. **Splitting Data**
   - Data dipisahkan menjadi 2 bagian (training dan testing) dengan 70% data training dan 30% data testing.
9. **Preprocesing pada kolom kategorikal**
    - Karena tipe data kategorikal tidak dapat diproses secara langsung, maka dilakukan OneHotEncoder.
10. **Preprocessing pada kolom numerikal**
    - Sedangkan pada kolom numerik, dilakukan preprocessing tanpa scalling.
11. **Membuat pipeline**
    - Untuk mempermudah pembuatan model, dibuat pipeline menggunakan GaussianNB.
12. **Melatih model**
    - Setelah membuat pipeline, maka model akan dilatih menggunakan predictor dan target dataset pelatihan.
13. **Evaluasi model**
    - Di tahap ini kita melakukan evaluasi dengan menghitung precision, recall, f1-score, dan confussion matrix dari hasil training.
14. **Visualisasi confussion matrix**
    - Ini dilakukan untuk peninjauan saja.
15. **Evaluasi menggunakan ROC Curve**
    - Ditujukan untuk mengukur performa klasifikasi dari model.
16. **Cross-Validation**
    - Di tahap ini dilakukan pengukuran apakah model akan bagus untuk hasil data baru atau tidak.
17. **Mengambil model terbaik dan mengevaluasinya kembali**
    - Setelah cross-validation, diambil data terbaik dari hasil grid search dan dilakukan evaliasi dengan cross validation kembali.
18. **Membuat sample prediksi**
    - Setelah itu, kita membuat 3 sample baru untuk dilakukan prediksi menggunakan model terbaik.
19. **Prediksi**
    - Setelah itu, sample tersebut diprediksi apakah layak atau tidak mendapatkan kredit komputer.
20. **Feature Importance**
    - Mengukur seberapa penting setiap fitur dari model terbaik yang diambil serta menyimpan model.
