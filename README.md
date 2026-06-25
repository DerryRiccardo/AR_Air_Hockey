# AR Air Hockey - Computer Vision Project

AR Air Hockey adalah project Computer Vision yang mengubah gesture tangan menjadi kontrol permainan air hockey. Sistem membaca tangan dari webcam, mendeteksi landmark dengan MediaPipe Hands, mengubah landmark menjadi fitur numerik, lalu memakai model machine learning untuk menentukan apakah paddle harus aktif atau diam.

Project ini memiliki dua cara menjalankan aplikasi:

- `air_hockey.py` untuk versi desktop berbasis OpenCV window.
- `app.py` untuk versi web app berbasis Flask dengan stream video di browser.

## Project Structure

```text
.
|-- prepare_dataset.py
|-- train_model.py
|-- test.py
|-- air_hockey.py
|-- app.py
|-- requirements.txt
|-- README.md
|-- project_explanation.md
|-- data/
|   |-- hand-gestures.csv
|   `-- gesture_data.csv
`-- model/
    |-- gesture_model.pkl
    |-- scaler.pkl
    |-- confusion_matrix.png
    |-- feature_importance.png
    `-- training_curve.png
```

## Main Workflow

1. `prepare_dataset.py`
   Membersihkan dataset gesture mentah dari Zenodo, menghapus fitur yang tidak informatif, mengecek keseimbangan kelas, lalu menyimpan `data/gesture_data.csv`.

2. `train_model.py`
   Melatih dua model kandidat, yaitu Random Forest dan SVM RBF, lalu memilih model terbaik berdasarkan 5-fold cross-validation. Output utamanya adalah `model/gesture_model.pkl` dan `model/scaler.pkl`.

3. `test.py`
   Menjalankan smoke test untuk memastikan dependency, kamera, model, dan pipeline inference siap dipakai.

4. `air_hockey.py`
   Menjalankan game desktop real-time dengan OpenCV.

5. `app.py`
   Menjalankan web app Flask. Game loop tetap berjalan di background thread, lalu frame hasil render dikirim ke browser sebagai MJPEG stream melalui endpoint `/video_feed`.

## How It Works

### 1. Webcam Input

Webcam menangkap frame video secara real-time. Frame ini menjadi input awal untuk seluruh pipeline.

### 2. Hand Detection with MediaPipe

MediaPipe Hands mendeteksi hingga dua tangan dan menghasilkan 21 landmark untuk setiap tangan. Landmark 8 mewakili ujung jari telunjuk, yang dipakai sebagai dasar posisi paddle.

### 3. Feature Extraction

Koordinat landmark tidak langsung dipakai mentah. Sistem menghitung jarak Euclidean setiap landmark ke wrist, lalu menormalisasinya. Setelah `dist_0` dibuang, tersisa 20 fitur yang konsisten dengan data training.

### 4. Gesture Classification

Model memprediksi dua kelas:

| Class | Gesture | Efek di Game |
|---|---|---|
| `0` | No pointing / closed fist | Paddle diam |
| `1` | Pointing / index finger up | Paddle mengikuti ujung jari |

### 5. Paddle Control and Game Logic

Jika gesture aktif, paddle diperbarui berdasarkan posisi fingertip. Player 1 berada di sisi kiri layar dan Player 2 di sisi kanan. Puck lalu bergerak menggunakan physics sederhana: wall bounce, collision, goal detection, rally boost, dan speed cap.

### 6. Rendering

- `air_hockey.py`: render langsung ke window OpenCV dengan `cv2.imshow`.
- `app.py`: render dikonversi menjadi JPEG dan di-stream ke browser dengan `multipart/x-mixed-replace`.

## Web App Mode

`app.py` menambahkan antarmuka browser untuk project ini. Arsitekturnya:

- Flask menangani route web.
- `game_loop()` berjalan di background thread.
- Frame terbaru disimpan di state global sebagai JPEG bytes.
- Browser mengambil stream dari `/video_feed`.
- Tombol Start, Stop, Restart memanggil endpoint `/start`, `/stop`, dan `/restart`.
- Endpoint `/status` dipakai front-end untuk menyinkronkan status game.

Ini membuat project lebih mudah didemokan tanpa perlu membuka window OpenCV secara langsung.

## Evaluation Summary

Model dilatih dengan split 80/20 dan 5-fold stratified cross-validation. Dari hasil training terbaru:

| Model | Test Accuracy | 5-Fold CV Mean | 5-Fold CV Std |
|---|---:|---:|---:|
| Random Forest | 0.9823 | 0.9825 | 0.0022 |
| SVM RBF | 0.9864 | 0.9860 | 0.0012 |

SVM RBF menjadi model aktif yang tersimpan di `model/gesture_model.pkl`.

Highlight metric yang penting:

- Accuracy tinggi karena dataset seimbang dan fiturnya cukup informatif.
- Precision tinggi berarti prediksi gesture aktif jarang salah alarm.
- Recall tinggi berarti gesture pointing jarang gagal dikenali.
- F1-score tinggi berarti precision dan recall sama-sama baik.
- Confusion matrix membantu melihat kesalahan per kelas, bukan hanya satu angka akurasi.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Pipeline

### 1. Prepare dataset

```bash
python prepare_dataset.py
```

### 2. Train model

```bash
python train_model.py
```

### 3. Run smoke test

```bash
python test.py
```

## Run the App

### Desktop version

```bash
python air_hockey.py
```

### Web app version

```bash
python app.py
```

Lalu buka:

```text
http://localhost:5000
```

## Controls

- Angkat jari telunjuk untuk menggerakkan paddle.
- Gesture non-pointing membuat paddle diam.
- Pada desktop app, tekan `Q` untuk keluar dan `R` untuk restart.
- Pada web app, gunakan tombol Start, Stop, dan Restart di browser.

