import logging
import os
import warnings
from uuid import uuid4
from pathlib import Path
from typing import TypedDict

from inference import predict_probabilities

#  【設定】TensorFlowのログ（oneDNNなどの通知）を抑制
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 警告以外は表示しない
import keras
import numpy as np
from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template,
    send_from_directory,
)
from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

# ===== ロギング設定 (チーム開発仕様) =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
# ディレクトリ構成の変更
DATA_DIA = BASE_DIR / "data"
UPLOAD_FOLDER = DATA_DIA / "uploads"
# 将来用（ログ保存など）
RESULTS_FOLDER = DATA_DIA / "results"
MODEL_PATH = BASE_DIR / "image_classifier.h5"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_REQUEST_SIZE = 6 * 1024 * 1024

MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096
MAX_IMAGE_PIXELS = 16_000_000


class ImageFormatRule(TypedDict):
    extensions: frozenset[str]
    mime_type: str


IMAGE_FORMAT_RULES: dict[str, ImageFormatRule] = {
    "PNG": {
        "extensions": frozenset({".png"}),
        "mime_type": "image/png",
    },
    "JPEG": {
        "extensions": frozenset({".jpg", ".jpeg"}),
        "mime_type": "image/jpeg",
    },
}

ALLOWED_EXTENSIONS = frozenset(
    extension
    for rule in IMAGE_FORMAT_RULES.values()
    for extension in rule["extensions"]
)

ALLOWED_MIME_TYPES = frozenset(
    rule["mime_type"] for rule in IMAGE_FORMAT_RULES.values()
)

LABELS = ["飛行機", "自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"]
RESULT_COUNT = 3
TTA_CROP_RATIOS = (1.0,)

app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER=str(UPLOAD_FOLDER),
    MAX_CONTENT_LENGTH=MAX_REQUEST_SIZE,
)

# ===== 遅延ロード（起動を止めない）=====
model = None
model_error = None


class UploadValidationError(Exception):
    """ユーザーへ安全に返せるアップロード検証エラー。"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


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
        logger.info("Model loaded successfully: %s", MODEL_PATH)
        logger.info(
            "Model shape: input=%s output=%s",
            model.input_shape,
            model.output_shape,
        )
        return model

    except Exception as e:
        model_error = e
        # エラーログも英語で詳細に出力
        logger.error(f"Failed to load model: {repr(e)}")
        raise


def get_file_extension(filename):
    return Path(filename).suffix.lower()


def is_allowed_extension(filename: str) -> bool:
    return get_file_extension(filename) in ALLOWED_EXTENSIONS


def get_uploaded_file_size(uploaded_file: FileStorage) -> int:
    try:
        uploaded_file.stream.seek(0, os.SEEK_END)
        file_size = uploaded_file.stream.tell()
        return file_size
    finally:
        uploaded_file.stream.seek(0)


def validate_file_size(uploaded_file: FileStorage) -> None:
    file_size = get_uploaded_file_size(uploaded_file)

    if file_size <= 0:
        raise UploadValidationError(
            "空のファイルはアップロードできません。",
            status_code=400,
        )
    if file_size > MAX_FILE_SIZE:
        raise UploadValidationError(
            "ファイルサイズは5MB以下にしてください。",
            status_code=413,
        )


def validate_declared_mime_type(uploaded_file: FileStorage) -> str:
    declared_mime_type = (uploaded_file.mimetype or "").lower()

    if declared_mime_type not in ALLOWED_MIME_TYPES:
        raise UploadValidationError(
            "PNGまたはJPEG画像を選択してください。",
            status_code=415,
        )
    return declared_mime_type


def validate_detected_image_format(
    filename: str,
    declared_mime_type: str,
    detected_format: str | None,
) -> None:
    normalized_format = (detected_format or "").upper()
    format_rule = IMAGE_FORMAT_RULES.get(normalized_format)

    if format_rule is None:
        raise UploadValidationError(
            "対応していない画像形式です。",
            status_code=415,
        )

    extension = get_file_extension(filename)

    if extension not in format_rule["extensions"]:
        raise UploadValidationError(
            "ファイルの拡張子と画像形式が一致しません。",
            status_code=415,
        )
    normalized_mime_type = declared_mime_type.split(";", 1)[0].strip().lower()

    if normalized_mime_type != format_rule["mime_type"]:
        raise UploadValidationError(
            "MIMEタイプと画像形式が一致しません。",
            status_code=415,
        )


def validate_image_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise UploadValidationError(
            "画像サイズが不正です。",
            status_code=400,
        )
    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
        raise UploadValidationError(
            "画像の縦横サイズが上限を超えています。",
            status_code=400,
        )
    if width * height > MAX_IMAGE_PIXELS:
        raise UploadValidationError(
            "画像の総ピクセル数が上限を超えています。",
            status_code=400,
        )


def validate_image_content(
    uploaded_file: FileStorage,
    declared_mime_type: str,
) -> None:
    detected_format = None

    try:
        with warnings.catch_warnings():
            # 巨大画像によるメモリ消費を通常のエラーとして処理する
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            uploaded_file.stream.seek(0)

            with Image.open(
                uploaded_file.stream,
                formats=("PNG", "JPEG"),
            ) as image:
                detected_format = image.format
                width, height = image.size

                validate_detected_image_format(
                    uploaded_file.filename or "",
                    declared_mime_type,
                    detected_format,
                )
                validate_image_dimensions(width, height)

                image.verify()

            uploaded_file.stream.seek(0)
            with Image.open(
                uploaded_file.stream,
                formats=("PNG", "JPEG"),
            ) as image:
                validate_image_dimensions(*image.size)

                image.load()
    except UploadValidationError:
        raise

    except (
        UnidentifiedImageError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
        OSError,
        ValueError,
    ) as error:
        logger.warning(
            "Invalid image content: filename=%s error=%s",
            secure_filename(uploaded_file.filename or ""),
            type(error).__name__,
        )
        raise UploadValidationError(
            "有効なPNGまたはJPEG画像ではありません。",
            status_code=400,
        ) from error

    finally:
        uploaded_file.stream.seek(0)


def validate_uploaded_file(uploaded_file: FileStorage) -> None:
    if not uploaded_file.filename:
        raise UploadValidationError(
            "画像ファイルを選択してください。",
            status_code=400,
        )
    if not is_allowed_extension(uploaded_file.filename):
        raise UploadValidationError(
            "PNGまたはJPEG画像を選択してください。",
            status_code=415,
        )
    validate_file_size(uploaded_file)

    declared_mime_type = validate_declared_mime_type(uploaded_file)

    validate_image_content(
        uploaded_file,
        declared_mime_type,
    )


def save_validated_upload(
    uploaded_file: FileStorage,
    upload_folder: Path,
) -> tuple[str, Path]:
    """
    検証済み画像をUUID名で保存する
    この関数を呼ぶ前にvalidate_uploaded_file()を実行する。
    """
    original_filename = uploaded_file.filename or ""
    extension = get_file_extension(original_filename)

    if extension not in ALLOWED_EXTENSIONS:
        raise RuntimeError("Validated upload has an unexpected extension")

    upload_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    stored_filename = f"{uuid4().hex}{extension}"
    filepath = upload_folder / stored_filename

    try:
        uploaded_file.stream.seek(0)
        uploaded_file.save(filepath)

    except Exception:
        filepath.unlink(missing_ok=True)
        raise

    return stored_filename, filepath


def predict_image(filepath: str | Path) -> list[dict[str, float | str]]:
    averaged_prediction = predict_probabilities(
        filepath,
        model=get_model(),
        crop_ratios=TTA_CROP_RATIOS,
        class_count=len(LABELS),
    )

    logger.info(
        "Inference completed: crop_ratios=%s model_calls=%d",
        TTA_CROP_RATIOS,
        len(TTA_CROP_RATIOS),
    )

    sorted_indices = np.argsort(averaged_prediction)[::-1]
    predictions = []

    for index in sorted_indices[:RESULT_COUNT]:
        label_index = int(index)
        probability = float(averaged_prediction[label_index])

        predictions.append(
            {
                "label": LABELS[label_index],
                "probability": round(probability * 100, 1),
            }
        )

    return predictions


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    logger.warning("Upload request exceeded maximum content length")
    return render_template(
        "index.html",
        error_message="ファイルサイズは5MB以下にしてください。",
    ), 413


# data/uploads 画像を表示するため
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    if "file" not in request.files:
        logger.warning("No file part in request")
        return render_template(
            "index.html",
            error_message="画像ファイルを選択してください。",
        ), 400

    uploaded_file = request.files["file"]

    try:
        validate_uploaded_file(uploaded_file)

    except UploadValidationError as error:
        logger.warning(
            "Upload validation failed: status=%s message=%s",
            error.status_code,
            error.message,
        )
        return render_template(
            "index.html",
            error_message=error.message,
        ), error.status_code

    uploaded_folder = Path(app.config["UPLOAD_FOLDER"])
    filepath: Path | None = None
    stored_filename: str | None = None

    try:
        stored_filename, filepath = save_validated_upload(
            uploaded_file,
            uploaded_folder,
        )
        logger.info(
            "File uploaded successfully: filename=%s mime=%s",
            stored_filename,
            uploaded_file.mimetype,
        )

        predictions = predict_image(filepath)

    except Exception:
        logger.exception(
            "Image classification failed: filename=%s",
            stored_filename or "not-saved",
        )
        if filepath is not None:
            filepath.unlink(missing_ok=True)

        return render_template(
            "index.html",
            error_message="画像の判定中にエラーが発生しました。",
        ), 500

    image_url = url_for(
        "uploaded_file",
        filename=stored_filename,
    )
    return render_template(
        "result.html",
        predictions=predictions,
        filepath=image_url,
    )


if __name__ == "__main__":
    logger.info("Starting Flask development server")
    app.run(debug=True, port=5000)

# コード内のコメント: 日本語でOK（日本人チームなら理解しやすさ優先）。

# ログ出力（printなど）: 英語推奨（文字化け防止、検索性向上）。

# ユーザーの画面: 日本語（ターゲットユーザーに合わせる）
