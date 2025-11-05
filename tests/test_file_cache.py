import io
import json
from pathlib import Path

import pytest

import s2aff.file_cache as file_cache


def test_cached_path_returns_existing_file(tmp_path):
    existing = tmp_path / "stored.txt"
    existing.write_text("hello")

    result = file_cache.cached_path(str(existing), cache_dir=str(tmp_path))

    assert result == str(existing)


def test_cached_path_missing_file_raises(tmp_path):
    missing = tmp_path / "missing.txt"
    with pytest.raises((FileNotFoundError, ValueError)):
        file_cache.cached_path(str(missing), cache_dir=str(tmp_path))


def test_get_from_cache_downloads_and_reuses(monkeypatch, tmp_path):
    url = "https://example.com/resource.bin"
    etag_value = "etag-value"
    calls = {"http_get": 0}

    class DummyHeadResponse:
        status_code = 200

        def __init__(self, headers):
            self.headers = headers

    def fake_head(requested_url, allow_redirects=True):
        assert requested_url == url
        return DummyHeadResponse({"ETag": etag_value})

    def fake_http_get(requested_url, temp_file):
        assert requested_url == url
        calls["http_get"] += 1
        temp_file.write(b"payload")

    monkeypatch.setattr(file_cache.requests, "head", fake_head)
    monkeypatch.setattr(file_cache, "http_get", fake_http_get)

    cache_path = file_cache.get_from_cache(url, cache_dir=str(tmp_path))
    cache_file = Path(cache_path)
    meta_file = Path(cache_path + ".json")

    assert cache_file.read_bytes() == b"payload"
    metadata = json.loads(meta_file.read_text())
    assert metadata == {"url": url, "etag": etag_value}

    second_path = file_cache.get_from_cache(url, cache_dir=str(tmp_path))
    assert second_path == cache_path
    assert calls["http_get"] == 1


def test_get_from_cache_head_failure_raises(monkeypatch, tmp_path):
    url = "https://example.com/bad.bin"

    class DummyHeadResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.headers = {}

    def fake_head(requested_url, allow_redirects=True):
        assert requested_url == url
        assert allow_redirects is True
        return DummyHeadResponse(503)

    monkeypatch.setattr(file_cache.requests, "head", fake_head)

    with pytest.raises(IOError):
        file_cache.get_from_cache(url, cache_dir=str(tmp_path))


def test_filename_to_url_requires_metadata(tmp_path):
    filename = "artifact.bin"
    file_path = tmp_path / filename
    file_path.write_text("data")

    with pytest.raises(FileNotFoundError):
        file_cache.filename_to_url(filename, cache_dir=str(tmp_path))


def test_cached_path_rejects_unknown_scheme(tmp_path):
    with pytest.raises(ValueError):
        file_cache.cached_path("s3://bucket/key", cache_dir=str(tmp_path))


def test_http_get_writes_non_empty_chunks(monkeypatch):
    chunks = [b"a", b"", b"bc"]

    class DummyGetResponse:
        def iter_content(self, chunk_size):
            assert chunk_size == 1024
            return iter(chunks)

    monkeypatch.setattr(file_cache.requests, "get", lambda url, stream=True: DummyGetResponse())

    buffer = io.BytesIO()
    file_cache.http_get("https://example.com/file.bin", buffer)

    assert buffer.getvalue() == b"abc"
