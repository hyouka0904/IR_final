"""src/download.py against a local HTTP server: resume, size checks, corrupt archives."""
import functools
import http.server
import io
import tarfile
import threading

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
