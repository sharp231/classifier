from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from uuid import UUID

import app as app_model


@pytest.fixture
def client():
    app_model.app.config["TESTING"] = True

    with app_model.app.test_client() as test_client:
        yield test_client


def test_homepage_loads(client):
    response = client.get("/")

    assert response.status_code == 200


def test_result_redirects_without_file(client):
    response = client.post("/result")

    assert response.status_code == 400
    assert "画像ファイルを選択してください。".encode() in response.data


def test_result_redirects_invalid_extension(client):
    data = {"file": (BytesIO(b"not an image"), "test.txt")}
    response = client.post("/result", data=data, content_type="multipart/form-data")

    assert response.status_code == 415
    assert "PNGまたはJPEG画像を選択してください。".encode() in response.data


def create_test_image():
    img = Image.new("RGB", (32, 32), color="red")
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


# 本物のモデル読み込みを避け、/result の画面表示までを安定してテストする
class DummyModel:
    def predict(self, x, verbose=0):
        return np.array([[0.9, 0.05, 0.03, 0.01, 0.01, 0, 0, 0, 0, 0]])


def test_result_uploads_image_successfully(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_model, "UPLOAD_FOLDER", tmp_path)
    app_model.app.config["UPLOAD_FOLDER"] = str(tmp_path)
    monkeypatch.setattr(app_model, "get_model", lambda: DummyModel())

    data = {"file": (create_test_image(), "test.png")}

    response = client.post(
        "/result",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"90.0" in response.data

    saved_files = list(tmp_path.glob("*.png"))

    assert len(saved_files) == 1
    saved_file = saved_files[0]

    assert saved_file.name != "test.png"
    assert f"/uploads/{saved_file.name}".encode() in response.data

    UUID(saved_file.stem)


def test_result_rejects_file_larger_than_limit(client):
    oversized_file = BytesIO(b"x" * (app_model.MAX_FILE_SIZE + 1))
    data = {"file": (oversized_file, "large.png")}

    response = client.post(
        "/result",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert "ファイルサイズは5MB以下にしてください".encode() in response.data


class FailingModel:
    def predict(self, *_args, **_kwargs):
        raise RuntimeError("simulated prediction failure")


def test_result_does_not_save_invalid_image(
    client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setitem(
        app_model.app.config,
        "UPLOAD_FOLDER",
        str(tmp_path),
    )
    response = client.post(
        "/result",
        data={
            "file": (
                BytesIO(b"not an image"),
                "fake.png",
                "image/png",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_result_removes_upload_when_prediction_fails(
    client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setitem(
        app_model.app.config,
        "UPLOAD_FOLDER",
        str(tmp_path),
    )
    monkeypatch.setattr(
        app_model,
        "get_model",
        lambda: FailingModel(),
    )
    response = client.post(
        "/result",
        data={
            "file": (
                create_test_image(),
                "sample.png",
                "image/png",
            )
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 500
    assert "画像の判定中にエラーが発生しました。".encode() in response.data
    assert list(tmp_path.iterdir()) == []
