from flask import Flask, request, render_template, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
import uuid
from datetime import datetime
import requests
import cloudinary
import cloudinary.uploader
import cloudinary.api

app = Flask(__name__)
app.secret_key = 'ayam-classifier-secret-key'  # Untuk flash messages

# Konfigurasi Cloudinary
cloudinary.config(
    cloud_name="dbrsonwrw",
    api_key="351131324987282",
    api_secret="1wJ5tZ5esCcix_RgS7OFZY2reOs",
    secure=True
)

# URL model di Google Drive
model_url = 'https://drive.google.com/uc?id=1n0pQ3Sz3TUhuLf7ie_r3OnY3AYwZ-4Bn'
model_path = 'mobilenet_chicken_model_v2_finetuned.h5'

# Download model jika belum ada
if not os.path.exists(model_path):
    print("🔽 Model tidak ditemukan, mendownload...")
    response = requests.get(model_url)
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("✅ Model berhasil di-download.")
    else:
        print("❌ Gagal mendownload model.")
        raise Exception("Gagal mendownload model dari Google Drive.")

# Load model
model = load_model(model_path)

# Kelas prediksi
class_names = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
history = []  # Menyimpan riwayat prediksi

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE_MB = 5

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def file_too_large(file):
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    return size > MAX_FILE_SIZE_MB * 1024 * 1024

@app.route('/')
def home():
    return render_template('index.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('file')
    predictions = []

    if not files:
        flash("Tidak ada file yang dipilih.", "danger")
        return redirect(url_for('home'))

    for file in files:
        if file and allowed_file(file.filename):
            if file_too_large(file):
                flash(f"File {file.filename} terlalu besar. Maksimum 5MB.", "warning")
                return redirect(url_for('home'))

            try:
                # Upload file ke Cloudinary
                upload_result = cloudinary.uploader.upload(file, folder="ayam-classification")
                image_url = upload_result['secure_url']

                # Reset pointer file
                file.seek(0)

                # 🔥 Ini yang diperbaiki: baca file ke BytesIO
                file_bytes = io.BytesIO(file.read())
                img = image.load_img(file_bytes, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                prediction = model.predict(img_array)[0]
                class_index = np.argmax(prediction)
                result = class_names[class_index]
                result_probs = {class_names[i]: f"{prediction[i]*100:.2f}%" for i in range(len(class_names))}

                predictions.append({
                    'filename': image_url,
                    'result': result,
                    'probs': result_probs,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                history.insert(0, predictions[-1])

            except Exception as e:
                flash(f"❌ Terjadi kesalahan saat memproses gambar: {e}", "danger")
                return redirect(url_for('home'))
        else:
            flash(f"File {file.filename} tidak didukung formatnya.", "danger")
            return redirect(url_for('home'))

    flash(f"Berhasil memproses {len(predictions)} gambar.", "success")
    return render_template('index.html', predictions=predictions, history=history)