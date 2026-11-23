"""Resumable downloads with a check-and-repair pass, standard library only (Windows / Linux / macOS).

Every archive goes through ensure_archive(), which first looks at what is already on disk:
- complete (same size as the remote file): kept
- a `.part` left by an interrupted run: resumed (FTP REST / HTTP Range; restarted if the server
  cannot resume)
- a finished file smaller than the remote one (left by an older, non-checking run): renamed back
  to `.part` and resumed; larger than the remote one: deleted and downloaded again
Extraction that fails, or finds no image in the archive (an all-zero file reads as an empty tar),
deletes the archive and downloads it once more. check_only=True reports without touching files.
"""
import ftplib
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

TIMEOUT = 60  # seconds without data before giving up (a firewall that drops FTP would hang forever)
JPEG_EOI = b"\xff\xd9"


class Progress:
    """One line per 5 %; every 50 MB when the total size is unknown."""

    def __init__(self, done: int = 0):
        self.total, self.done, self.last = 0, done, -1

    def add(self, n: int) -> None:
        self.done += n
        step = self.done * 100 // self.total // 5 if self.total else self.done // (50 << 20)
        if step != self.last:
            self.last = step
            mb = f"   {self.done / 2**20:,.0f} MB"
            print(mb + (f" / {self.total / 2**20:,.0f} MB ({self.done * 100 // self.total}%)"
                        if self.total else ""), flush=True)


def _fetch_http(url: str, part: Path, progress: Progress) -> None:
    req = urllib.request.Request(url)
    if progress.done:
        req.add_header("Range", f"bytes={progress.done}-")
    try:
        r = urllib.request.urlopen(req, timeout=TIMEOUT)
    except urllib.error.HTTPError as e:
        if e.code != 416 or not progress.done:
            raise
        part.unlink()  # 416: the .part is not smaller than the file; its content is unknown
        progress.done = 0
        r = urllib.request.urlopen(url, timeout=TIMEOUT)
    with r:
        if progress.done and r.status != 206:  # the server ignored Range: start over
            progress.done = 0
        length = r.headers.get("Content-Length")
        progress.total = progress.done + int(length) if length else 0
        with open(part, "ab" if progress.done else "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                progress.add(len(chunk))


def _ftp_connect(url: str) -> tuple[ftplib.FTP, str]:
    u = urllib.parse.urlparse(url)
    ftp = ftplib.FTP(timeout=TIMEOUT)  # unlike urllib, the timeout covers the data connection too
    ftp.connect(u.hostname, u.port or 21)
    ftp.login()
    ftp.voidcmd("TYPE I")
    return ftp, u.path


def _fetch_ftp(url: str, part: Path, progress: Progress) -> None:
    ftp, path = _ftp_connect(url)
    with ftp:
        progress.total = ftp.size(path) or 0
        if progress.total and progress.done >= progress.total:  # the .part is already complete
            return
        print("   connected, transferring", flush=True)
        with open(part, "ab" if progress.done else "wb") as f:
            def write(chunk: bytes) -> None:
                f.write(chunk)
                progress.add(len(chunk))

            ftp.retrbinary(f"RETR {path}", write, blocksize=1 << 16, rest=progress.done or None)


def fetch(url: str, dst: Path) -> Path:
    """Download url to dst (skipped if dst exists and is non-empty), resuming a `.part`."""
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    print(f">> download {url}" + (f" (resuming at {offset / 2**20:,.0f} MB)" if offset else ""),
          flush=True)
    progress = Progress(offset)
    if url.startswith("ftp://"):
        _fetch_ftp(url, part, progress)
    else:
        _fetch_http(url, part, progress)
    # a connection that ends early without an error leaves a short file: keep it as .part
    if progress.total and part.stat().st_size != progress.total:
        raise OSError(f"{dst.name} is incomplete ({part.stat().st_size:,} / {progress.total:,} "
                      f"bytes); run again to resume")
    os.replace(part, dst)
    return dst


def remote_size(url: str) -> int | None:
    """Size of the remote file, or None if the server can't be reached or doesn't say."""
    try:
        if url.startswith("ftp://"):
            ftp, path = _ftp_connect(url)
            with ftp:
                return ftp.size(path)
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"),
                                    timeout=TIMEOUT) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except (OSError, EOFError, ftplib.Error, ValueError):
        return None


def _mb(n: int) -> str:
    return f"{n / 2**20:,.0f} MB"


def ensure_archive(url: str, dst: Path, check_only: bool = False) -> Path | None:
    """Make dst a complete copy of url (see the module docstring). Returns None in check_only mode
    when the file is not usable yet."""
    dst = Path(dst)
    part = dst.with_name(dst.name + ".part")
    if dst.exists():
        size, local = remote_size(url), dst.stat().st_size
        if size is None or local == size:
            return dst
        if local < size:
            print(f"   {dst.name}: {_mb(local)}, smaller than the remote {_mb(size)} (interrupted); "
                  + ("can resume" if check_only else "resuming"))
            if check_only:
                return None
            part.unlink(missing_ok=True)
            os.replace(dst, part)
        else:
            print(f"   {dst.name}: larger than the remote file; "
                  + ("needs a fresh download" if check_only else "downloading again"))
            if check_only:
                return None
            dst.unlink()
    elif part.exists():
        print(f"   {dst.name}: partly downloaded ({_mb(part.stat().st_size)}); "
              + ("can resume" if check_only else "resuming"))
        if check_only:
            return None
    elif check_only:
        print(f"   {dst.name}: not downloaded")
        return None
    return fetch(url, dst)


def extract_jpgs(archive: Path, dst: Path) -> int:
    """Extract every .jpg of a tar archive flat into dst (some archives nest folders)."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(archive) as tar:
        for m in tar:
            if m.isfile() and m.name.lower().endswith(".jpg"):
                with tar.extractfile(m) as src, open(dst / Path(m.name).name, "wb") as out:
                    shutil.copyfileobj(src, out)
                n += 1
    if n == 0:  # an all-zero file reads as an empty tar without an error
        raise tarfile.ReadError("no .jpg in the archive")
    return n


def extract_with_retry(url: str, archive: Path, dst: Path) -> None:
    """A corrupt archive is deleted and downloaded once more; a second failure stops the run."""
    for attempt in (1, 2):
        try:
            print(f">> extract {archive.name}", flush=True)
            extract_jpgs(archive, dst)
            return
        except (tarfile.TarError, EOFError, OSError) as e:
            archive.unlink(missing_ok=True)
            if attempt == 2:
                sys.exit(f"!! {archive.name} still fails to extract after a fresh download "
                         f"({type(e).__name__}: {e}); try again later")
            print(f"!! {archive.name} failed to extract ({type(e).__name__}: {e}); "
                  f"deleted, downloading again")
            fetch(url, archive)


# ---------------- image checks ----------------

def image_problem(path, full: bool = False) -> str | None:
    """None if the image is fine, else a short reason ("missing", "empty", "unreadable: ...").

    The fast check opens the header and, for JPEG, looks for the end-of-image marker that a
    truncated download or extraction lacks (decoding fully to confirm, since a few valid JPEGs
    carry trailing bytes). full=True decodes every image.
    """
    path = str(path)
    if not os.path.exists(path):
        return "missing"
    size = os.path.getsize(path)
    if size == 0:
        return "empty"
    try:
        with Image.open(path) as im:
            fmt = im.format
            if im.width == 0 or im.height == 0:
                return "zero size"
        decode = full
        if fmt == "JPEG" and not full:
            with open(path, "rb") as f:
                f.seek(max(0, size - 64))
                decode = not f.read().rstrip(b"\x00\r\n ").endswith(JPEG_EOI)
        if decode:
            with Image.open(path) as im:
                im.load()
    except Exception as e:  # PIL raises many different types for damaged files
        return f"unreadable ({type(e).__name__}: {e})"
    return None


def image_problems(paths: list, full: bool = False, workers: int = 8) -> list[str | None]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: image_problem(p, full), paths))


def split_problems(paths: list, full: bool = False) -> tuple[list[Path], list[Path]]:
    """(missing, broken) among paths."""
    problems = image_problems(paths, full)
    missing = [Path(p) for p, m in zip(paths, problems) if m == "missing"]
    broken = [Path(p) for p, m in zip(paths, problems) if m and m != "missing"]
    return missing, broken
