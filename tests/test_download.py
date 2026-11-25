"""src/download.py against a local HTTP server: resume, size checks, corrupt archives."""
import functools
import http.server
import io
import tarfile
import threading
import urllib.error

import pytest
from PIL import Image

from src import download


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves files with HEAD, Range and 416 support."""

    def log_message(self, *a):
        pass

    def _file(self):
        return self.server.root / self.path.lstrip("/")

    def do_HEAD(self):
        f = self._file()
        self.send_response(200)
        self.send_header("Content-Length", str(f.stat().st_size))
        self.end_headers()

    def do_GET(self):
        if not self._file().exists():
            self.send_error(404)
            return
        data = self._file().read_bytes()
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= len(data):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
            data = data[start:]
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "srv"
    root.mkdir()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.root = root
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _jpeg(color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(buf, "JPEG")
    return buf.getvalue()


def _tgz(path, names):
    with tarfile.open(path, "w:gz") as t:
        for n in names:
            data = _jpeg()
            info = tarfile.TarInfo(f"nested/dir/{n}.jpg")
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))


def test_fetch_resumes_a_part(server, tmp_path):
    root, base = server
    payload = bytes(range(256)) * 4000
    (root / "f.bin").write_bytes(payload)
    dst = tmp_path / "f.bin"
    dst.with_name("f.bin.part").write_bytes(payload[:12345])
    download.fetch(f"{base}/f.bin", dst)
    assert dst.read_bytes() == payload and not dst.with_name("f.bin.part").exists()


class _DroppingHandler(_Handler):
    """Sends only the first `cut` bytes of each response (Content-Length still says the full
    size), the first `drops` times; then serves normally."""

    def do_GET(self):
        srv = self.server
        if srv.drops <= 0:
            return super().do_GET()
        srv.drops -= 1
        data = self._file().read_bytes()
        start = int(self.headers["Range"].split("=")[1].rstrip("-")) if self.headers.get(
            "Range") else 0
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:start + srv.cut])
        self.wfile.flush()
        self.connection.shutdown(2)  # like a server that resets the connection mid-transfer


@pytest.fixture
def dropping_server(tmp_path):
    root = tmp_path / "srv_drop"
    root.mkdir()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _DroppingHandler)
    srv.root, srv.drops, srv.cut = root, 0, 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, root, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_fetch_resumes_after_dropped_connections(dropping_server, tmp_path, monkeypatch):
    srv, root, base = dropping_server
    monkeypatch.setattr(download, "RETRY_WAIT", 0)
    payload = bytes(range(256)) * 4000
    (root / "f.bin").write_bytes(payload)
    srv.drops, srv.cut = 3, 100_000
    dst = download.fetch(f"{base}/f.bin", tmp_path / "f.bin")
    assert dst.read_bytes() == payload and not dst.with_name("f.bin.part").exists()


def test_fetch_gives_up_but_keeps_the_part(dropping_server, tmp_path, monkeypatch):
    srv, root, base = dropping_server
    monkeypatch.setattr(download, "RETRY_WAIT", 0)
    monkeypatch.setattr(download, "RETRIES", 2)
    (root / "f.bin").write_bytes(b"z" * 500_000)
    srv.drops, srv.cut = 10, 50_000
    with pytest.raises(download.IncompleteDownload, match="run again to resume"):
        download.fetch(f"{base}/f.bin", tmp_path / "f.bin")
    assert (tmp_path / "f.bin.part").stat().st_size == 150_000  # 3 attempts x 50 kB
    assert not (tmp_path / "f.bin").exists()


def test_fetch_does_not_retry_not_found(server, tmp_path, monkeypatch):
    _, base = server
    calls = []
    monkeypatch.setattr(download.time, "sleep", lambda s: calls.append(s))
    with pytest.raises(urllib.error.HTTPError):
        download.fetch(f"{base}/missing.bin", tmp_path / "m.bin")
    assert calls == []


def test_ensure_archive_short_file_resumes_and_long_file_restarts(server, tmp_path):
    root, base = server
    payload = b"abc" * 50000
    (root / "a.bin").write_bytes(payload)
    short = tmp_path / "short.bin"
    short.write_bytes(payload[:1000])  # renamed although incomplete (older script)
    assert download.ensure_archive(f"{base}/a.bin", short).read_bytes() == payload
    long = tmp_path / "long.bin"
    long.write_bytes(payload + b"junk")
    assert download.ensure_archive(f"{base}/a.bin", long).read_bytes() == payload


def test_check_only_changes_nothing(server, tmp_path):
    root, base = server
    (root / "a.bin").write_bytes(b"x" * 5000)
    part = tmp_path / "a.bin.part"
    part.write_bytes(b"x" * 10)
    assert download.ensure_archive(f"{base}/a.bin", tmp_path / "a.bin", check_only=True) is None
    assert part.read_bytes() == b"x" * 10 and not (tmp_path / "a.bin").exists()


def test_extract_flat_and_retry_on_corrupt_archive(server, tmp_path):
    root, base = server
    _tgz(root / "imgs.tgz", ["a", "b"])
    bad = tmp_path / "imgs.tgz"
    bad.write_bytes(b"\0" * (root / "imgs.tgz").stat().st_size)  # same size, reads as empty tar
    download.extract_with_retry(f"{base}/imgs.tgz", bad, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a.jpg", "b.jpg"]


def test_image_problem(tmp_path):
    good = tmp_path / "good.jpg"
    good.write_bytes(_jpeg())
    assert download.image_problem(good) is None
    cut = tmp_path / "cut.jpg"
    cut.write_bytes(good.read_bytes()[:60])
    assert download.image_problem(cut) is not None
    trailer = tmp_path / "trailer.jpg"
    trailer.write_bytes(good.read_bytes() + b"extra")  # valid JPEG with bytes after EOI
    assert download.image_problem(trailer) is None
    (tmp_path / "empty.jpg").write_bytes(b"")
    assert download.image_problem(tmp_path / "empty.jpg") == "empty"
    assert download.image_problem(tmp_path / "nope.jpg") == "missing"
    missing, broken = download.split_problems([good, cut, tmp_path / "nope.jpg"])
    assert missing == [tmp_path / "nope.jpg"] and broken == [cut]
