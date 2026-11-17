"""Daphnet Freezing of Gait 데이터셋을 UCI에서 받아 data/raw/에 푼다.

실행: .venv\\Scripts\\python.exe scripts\\download_data.py
이미 받아져 있으면 건너뛴다. (--dest 로 다른 폴더 지정 가능, 테스트용)
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip"
ROOT = Path(__file__).resolve().parents[1]
N_FILES = 17   # S01R01 ~ S10R01 run 파일 수


def extract(zf: zipfile.ZipFile, dest: Path):
    """zip 안에 zip이 또 들어 있는 경우도 처리."""
    for name in zf.namelist():
        if name.endswith(".zip"):
            extract(zipfile.ZipFile(io.BytesIO(zf.read(name))), dest)
    zf.extractall(dest, members=[n for n in zf.namelist() if not n.endswith(".zip")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=ROOT / "data" / "raw")
    dest = ap.parse_args().dest
    data_dir = dest / "dataset_fog_release" / "dataset"
    if len(list(data_dir.glob("S*R*.txt"))) == N_FILES:
        print(f"이미 있음: {data_dir}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"다운로드 중 (약 20MB): {URL}")
    with urllib.request.urlopen(URL, timeout=120) as r:
        blob = r.read()
    (dest / "daphnet.zip").write_bytes(blob)
    extract(zipfile.ZipFile(io.BytesIO(blob)), dest)

    n = len(list(data_dir.glob("S*R*.txt")))
    if n != N_FILES:
        raise SystemExit(f"파일 수가 다릅니다: {n}개 (기대 {N_FILES}개). {dest} 확인 필요")
    print(f"완료: {data_dir} ({n}개 파일)")


if __name__ == "__main__":
    main()
