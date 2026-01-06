import os
import logging

# 【設定】TensorFlowのログ（oneDNNなどの通知）を抑制
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 警告以外は表示しない

from flask import Flask, request, redirect, url_for, render_template,send_from_directory
from werkzeug.utils import secure_filename
import keras
import numpy as np
from PIL import Image, ImageOps

# ===== ロギング設定 (チーム開発仕様) =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ディレクトリ構成の変更
DATA_DIA = os.path.join(BASE_DIR,"data")
UPLOAD_FOLDER = os.path.join(DATA_DIA, "uploads")
# 将来用（ログ保存など）
RESULTS_FOLDER = os.path.join(DATA_DIA,"results")

MODEL_PATH = os.path.join(BASE_DIR, "image_classifier.h5")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels = ["飛行機","自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"]
img_size = 32
n_result = 3

app = Flask(__name__)

#フォルダがなければ自動作成
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
os.makedirs(RESULTS_FOLDER,exist_ok=True)

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
        # モデルのロード
        model = keras.saving.load_model(MODEL_PATH, compile=False)
        
        # 英語ログ出力
        logger.info(f"Model loaded successfully: {MODEL_PATH}")
        return model
    except Exception as e:
        model_error = e
        # エラーログも英語で詳細に出力
        logger.error(f"Failed to load model: {repr(e)}")
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

#data/uploads 画像を表示するため
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    if "file" not in request.files:
        logger.warning("No file part in request")
        return redirect(url_for("index"))

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        logger.warning(f"Invalid file or no selection: {file.filename}")
        return redirect(url_for("index"))

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    logger.info(f"File uploaded: {filename}")

    ratios = [1.0, 0.8, 0.65]
    preds = []

    m = get_model()

    # 複数パターンで推論（アンサンブル）
    for r in ratios:
        x, img32 = preprocess(filepath, img_size=32, crop_right_ratio=r)
        # デバッグ用の画像保存（必要なければ削除可）
        # img32.save(os.path.join(UPLOAD_FOLDER, f"_preprocessed_{str(r).replace('.','')}.png"))
        
        y_r = m.predict(x, verbose=0)[0]
        preds.append(y_r)

    # 結果の平均をとる
    y = np.mean(np.stack(preds, axis=0), axis=0)
    sorted_idx = np.argsort(y)[::-1]

    # 【変更】HTML文字列ではなく、データのリストを作成する
    predictions = []
    for i in range(n_result):
        idx = int(sorted_idx[i])
        ratio = float(y[idx])
        label = labels[idx]
        
        # 辞書型でリストに追加
        predictions.append({
            "label": label,
            "probability": round(ratio * 100, 1)
        })

    image_url = url_for("uploaded_file", filename=filename)
    
    # テンプレートに 'predictions' を渡す
    return render_template("result.html", predictions=predictions, filepath=image_url)

if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(debug=True, port=5000)

# コード内のコメント: 日本語でOK（日本人チームなら理解しやすさ優先）。

# ログ出力（printなど）: 英語推奨（文字化け防止、検索性向上）。

# ユーザーの画面: 日本語（ターゲットユーザーに合わせる）