from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
import os
import io
import zipfile
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    original = cv2.imread(file_path)
    resized = cv2.resize(original, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    padding = 20
    card_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 100 < w < 400 and 150 < h < 600:
            x_pad = max(x - padding, 0)
            y_pad = max(y - padding, 0)
            w_pad = w + 2 * padding
            h_pad = h + 2 * padding
            x_orig, y_orig = x_pad * 2, y_pad * 2
            w_orig, h_orig = w_pad * 2, h_pad * 2
            card_crop = original[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]
            card_images.append(card_crop)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zipf:
        for i, card in enumerate(card_images):
            _, buffer = cv2.imencode(".png", card)
            zipf.writestr(f"card_{i+1}.png", buffer.tobytes())

    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype="application/zip", as_attachment=True, download_name="cropped_cards.zip")

if __name__ == "__main__":
    app.run(debug=True)
