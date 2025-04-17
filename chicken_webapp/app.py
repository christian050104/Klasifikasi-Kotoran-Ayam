from flask import Flask, request, render_template, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
import uuid
from datetime import datetime
import requests

app = Flask(__name__)
app.secret_key = 'ayam-classifier-secret-key'  # Dibutuhkan untuk flash messages

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# URL model di Google Drive (ganti ID sesuai file kamu)
model_url = 'https://drive.google.com/uc?id=1n0pQ3Sz3TUhuLf7ie_r3OnY3AYwZ-4Bn'
model_path = 'mobilenet_chicken_model_v2_finetuned.h5'

# Download model jika belum ada
if not os.path.exists(model_path):
    print("🔽 Model tidak ditemukan, mendownload...")
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    print("✅ Model berhasil di-download.")

# Load model
model = load_model(model_path)

# Mapping index ke nama kelas
class_names = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
history = []  # Simpan hasil prediksi terbaru

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE_MB = 5

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def file_too_large(file):
    return len(file.read()) > MAX_FILE_SIZE_MB * 1024 * 1024

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
            file.seek(0)
            if file_too_large(file):
                flash(f"File {file.filename} terlalu besar. Maks 5MB.", "warning")
                return redirect(url_for('home'))
            file.seek(0)

            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            pred = model.predict(img_array)[0]
            class_index = np.argmax(pred)
            result = class_names[class_index]
            result_probs = {class_names[i]: f"{pred[i]*100:.2f}%" for i in range(len(class_names))}

            predictions.append({
                'filename': filename,
                'result': result,
                'probs': result_probs,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            history.insert(0, predictions[-1])
        else:
            flash(f"File {file.filename} tidak didukung.", "danger")
            return redirect(url_for('home'))

    flash(f"Berhasil memproses {len(predictions)} gambar.", "success")
    return render_template('index.html', predictions=predictions, history=history)

# Tidak perlu pakai webbrowser atau app.run, Railway akan handle lewat gunicorn
