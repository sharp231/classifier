import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from markupsafe import Markup
import keras
import numpy as np
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "images")
MODEL_PATH = os.path.join(BASE_DIR, "image_classifier.h5")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels = ["飛行機","自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"]
img_size = 32
n_result = 3

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ===== 遅延ロード（起動を止めない）=====
model = None
model_error = None

def get_model():
    global model, model_error
    if model is not None:
        return model
    if model_error is not None:
        raise model_error

    try:
        print("CWD:", os.getcwd())
        print("__file__:", os.path.abspath(__file__))
        print("MODEL_PATH:", MODEL_PATH)
        print("MODEL_EXISTS:", os.path.exists(MODEL_PATH))

        model = keras.saving.load_model(MODEL_PATH, compile=False)
        print("MODEL_LOADED:", model.input_shape, model.output_shape)
        return model
    except Exception as e:
        model_error = e
        print("MODEL_LOAD_FAILED:", repr(e))
        raise

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(filepath, img_size=32, crop_right_ratio=1.0):
    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    if crop_right_ratio < 1.0:
        w, h = img.size
        img = img.crop((0, 0, int(w * crop_right_ratio), h))

    img = ImageOps.fit(img, (img_size, img_size), method=Image.Resampling.LANCZOS)

    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, 0)
    return x, img

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return redirect(url_for("index"))

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    ratios = [1.0, 0.8, 0.65]
    preds = []

    m = get_model()

    for r in ratios:
        x, img32 = preprocess(filepath, img_size=32, crop_right_ratio=r)
        img32.save(os.path.join(UPLOAD_FOLDER, f"_preprocessed_{str(r).replace('.','')}.png"))
        y_r = m.predict(x, verbose=0)[0]
        preds.append(y_r)

    y = np.mean(np.stack(preds, axis=0), axis=0)
    sorted_idx = np.argsort(y)[::-1]

    result_html = ""
    for i in range(n_result):
        idx = int(sorted_idx[i])
        ratio = float(y[idx])
        label = labels[idx]
        result_html += f"<p>{round(ratio*100, 1)}%の確率で{label}です。</p>"

    # NOTE: filepathをimg srcに使うならURL化が必要（/static/...）
    image_url = url_for("static", filename=f"images/{filename}")
    return render_template("result.html", result=Markup(result_html), filepath=image_url)

if __name__ == "__main__":
    app.run(debug=True, port=5000)