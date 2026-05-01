from io import BytesIO

import numpy as np
import pytest
from PIL import Image

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

    assert response.status_code == 302
    assert response.location.endswith("/")


def test_result_redirects_invalid_extension(client):
    data = {"file": (BytesIO(b"not an image"), "test.txt")}
    response = client.post("/result", data=data, content_type="multipart/form-data")

    assert response.status_code == 302
    assert response.location.endswith("/")


def create_test_image():
    img = Image.new("RGB", (32, 32), color="red")
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


class DummyModel:
    def predict(self, x, verbose=0):
        return np.array([[0.9, 0.05, 0.03, 0.01, 0.01, 0, 0, 0, 0, 0]])


def test_result_uploads_image_successfully(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_model, "UPLOAD_FOLDER", str(tmp_path))
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
    assert b"/uploads/test.png" in response.data