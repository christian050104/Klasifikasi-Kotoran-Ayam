from flask import Flask, request, render_template, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
import io
import cloudinary
import cloudinary.uploader
import cloudinary.api
import gdown
from datetime import datetime
from pytz import timezone

app = Flask(__name__)
app.secret_key = 'ayam-classifier-secret-key'

# Konfigurasi Cloudinary
cloudinary.config(
    cloud_name="dbrsonwrw",
    api_key="351131324987282",
    api_secret="1wJ5tZ5esCcix_RgS7OFZY2reOs",
    secure=True
)

# Konfigurasi Model
MODEL_FILENAME = 'mobilenet_chicken_model_v2_finetuned_fix.keras'
GOOGLE_DRIVE_ID = '1-PHN7VLTxsMhXOquIY9Ni_KU6sLLmFpc'

# Download model jika belum ada
if not os.path.exists(MODEL_FILENAME):
    print("🔽 Model belum ada, mendownload dari Google Drive...")
    url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}'
    gdown.download(url, MODEL_FILENAME, quiet=False)
    print("✅ Model berhasil di-download!")

# Load model
model = load_model(MODEL_FILENAME)

# Nama kelas prediksi
class_names = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
history = []

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE_MB = 5

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def file_too_large(file):
    file.stream.seek(0, os.SEEK_END)
    size = file.stream.tell()
    file.stream.seek(0)
    return size > MAX_FILE_SIZE_MB * 1024 * 1024

@app.route('/')
def home():
    return render_template('index.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('file')
    predictions = []

    if not files:
        flash("❌ Tidak ada file yang dipilih.", "danger")
        return redirect(url_for('home'))

    for file in files:
        if file and allowed_file(file.filename):
            if file_too_large(file):
                flash(f"❌ File {file.filename} terlalu besar. Maksimum {MAX_FILE_SIZE_MB}MB.", "warning")
                return redirect(url_for('home'))

            try:
                # Upload ke Cloudinary
                upload_result = cloudinary.uploader.upload(file, folder="ayam-classification")
                image_url = upload_result['secure_url']

                # Reset pointer file
                file.stream.seek(0)

                # Load gambar dari file upload
                img = image.load_img(file.stream, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Prediksi
                preds = model.predict(img_array)[0]
                class_index = np.argmax(preds)
                result = class_names[class_index]
                result_probs = {class_names[i]: f"{preds[i]*100:.2f}%" for i in range(len(class_names))}

                # Waktu dalam format Indonesia (WIB)
                wib_time = datetime.now(timezone('Asia/Jakarta')).strftime('%d-%m-%Y %H:%M:%S WIB')

                predictions.append({
                    'filename': image_url,
                    'result': result,
                    'probs': result_probs,
                    'time': wib_time
                })
                history.insert(0, predictions[-1])

            except Exception as e:
                flash(f"❌ Terjadi kesalahan saat memproses gambar: {str(e)}", "danger")
                return redirect(url_for('home'))
        else:
            flash(f"❌ File {file.filename} tidak didukung formatnya.", "danger")
            return redirect(url_for('home'))

    flash(f"✅ Berhasil memproses {len(predictions)} gambar.", "success")
    return render_template('index.html', predictions=predictions, history=history)

# Railway pakai Gunicorn, jadi tidak perlu app.run()
