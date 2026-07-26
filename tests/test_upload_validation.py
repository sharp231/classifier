from io import BytesIO

import pytest
from PIL import Image
from werkzeug.datastructures import FileStorage
from uuid import UUID
from pathlib import Path

import app as app_model


def create_image_stream(image_format: str = "PNG") -> BytesIO:
    image_stream = BytesIO()

    Image.new(
        "RGB",
        (32, 32),
        color="red",
    ).save(
        image_stream,
        format=image_format,
    )

    image_stream.seek(0)
    return image_stream


def create_uploaded_file(
    stream: BytesIO,
    filename: str,
    content_type: str,
) -> FileStorage:
    return FileStorage(
        stream=stream,
        filename=filename,
        content_type=content_type,
    )


def test_accepts_valid_png():
    uploaded_file = create_uploaded_file(
        stream=create_image_stream("PNG"),
        filename="sample.png",
        content_type="image/png",
    )

    app_model.validate_uploaded_file(uploaded_file)


def test_rejects_fake_png_content():
    uploaded_file = create_uploaded_file(
        stream=BytesIO(b"not an image"),
        filename="sample.png",
        content_type="image/png",
    )

    with pytest.raises(app_model.UploadValidationError) as error_info:
        app_model.validate_uploaded_file(uploaded_file)

    assert error_info.value.status_code == 400
    assert error_info.value.message == "有効なPNGまたはJPEG画像ではありません。"


def test_rejects_extension_and_image_format_mismatch():
    uploaded_file = create_uploaded_file(
        stream=create_image_stream("PNG"),
        filename="sample.jpg",
        content_type="image/jpeg",
    )
    with pytest.raises(app_model.UploadValidationError) as error_info:
        app_model.validate_uploaded_file(uploaded_file)

    assert error_info.value.status_code == 415
    assert error_info.value.message == "ファイルの拡張子と画像形式が一致しません。"


def test_rejects_mime_and_image_format_mismatch():
    uploaded_file = create_uploaded_file(
        stream=create_image_stream("PNG"),
        filename="sample.png",
        content_type="image/jpeg",
    )
    with pytest.raises(app_model.UploadValidationError) as error_info:
        app_model.validate_uploaded_file(uploaded_file)

    assert error_info.value.status_code == 415
    assert error_info.value.message == "MIMEタイプと画像形式が一致しません。"


def test_saves_validated_upload_uuid_name(tmp_path):
    uploaded_file = create_uploaded_file(
        stream=create_image_stream("PNG"),
        filename="sample.png",
        content_type="image/png",
    )
    app_model.validate_uploaded_file(uploaded_file)

    sorted_filename, filepath = app_model.save_validated_upload(
        uploaded_file,
        tmp_path,
    )

    assert filepath.exists()
    assert filepath.parent == tmp_path
    assert filepath.name == sorted_filename
    assert sorted_filename != "sample.png"
    assert filepath.suffix == ".png"

    UUID(filepath.stem)


def test_removes_partial_file_when_save_fails(
    monkeypatch,
    tmp_path,
):
    uploaded_file = create_uploaded_file(
        stream=create_image_stream("PNG"),
        filename="sample.png",
        content_type="image/png",
    )
    app_model.validate_uploaded_file(uploaded_file)

    def failing_save(self, destination, buffer_size=16384):
        destination_path = Path(destination)
        destination_path.write_bytes(b"partial")
        raise OSError("simulated storage failure")

    monkeypatch.setattr(
        FileStorage,
        "save",
        failing_save,
    )
    with pytest.raises(OSError, match="simulated storage failure"):
        app_model.save_validated_upload(
            uploaded_file,
            tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_upload_validation_error_keeps_public_error_data():
    error = app_model.UploadValidationError(
        "テスト用エラーです。",
        status_code=415,
    )
    assert str(error) == "テスト用エラーです。"
    assert error.message == "テスト用エラーです。"
    assert error.status_code == 415


def test_rejects_empty_file():
    uploaded_file = create_uploaded_file(
        BytesIO(),
        "empty.png",
        "image/png",
    )
    with pytest.raises(app_model.UploadValidationError) as error:
        app_model.validate_uploaded_file(uploaded_file)

    assert error.value.status_code == 400


def test_rejects_unsupported_declared_mime_type():
    uploaded_file = create_uploaded_file(
        create_image_stream("PNG"),
        "sample.png",
        "application/octet-stream",
    )
    with pytest.raises(app_model.UploadValidationError) as error:
        app_model.validate_uploaded_file(uploaded_file)

    assert error.value.status_code == 415


def test_accepts_valid_jpeg():
    uploaded_file = create_uploaded_file(
        create_image_stream("JPEG"),
        "sample.jpg",
        "image/jpeg",
    )
    app_model.validate_uploaded_file(uploaded_file)


def test_rejects_image_over_dimension_limit(monkeypatch):
    monkeypatch.setattr(app_model, "MAX_IMAGE_WIDTH", 16)

    uploaded_file = create_uploaded_file(
        create_image_stream("PNG"),
        "sample.png",
        "image/png",
    )

    with pytest.raises(app_model.UploadValidationError) as error:
        app_model.validate_uploaded_file(uploaded_file)

    assert error.value.status_code == 400
