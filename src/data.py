"""Daphnet 데이터 로더 (여러 스크립트가 같이 사용)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "dataset_fog_release" / "dataset"
FS = 64  # Hz

COLUMNS = ["time_ms",
           "ankle_x", "ankle_y", "ankle_z",    # 발목(정강이)
           "thigh_x", "thigh_y", "thigh_z",    # 허벅지
           "trunk_x", "trunk_y", "trunk_z",    # 몸통(허리)
           "label"]                            # 0 실험 외 / 1 정상 / 2 동결
SENSORS = ["ankle", "thigh", "trunk"]
ACC_COLS = COLUMNS[1:10]


def load_run(subject: str, run: str) -> pd.DataFrame:
    """한 run 파일(예: S01, R01)을 읽는다."""
    df = pd.read_csv(DATA_DIR / f"{subject}{run}.txt", sep=" ", header=None, names=COLUMNS)
    df["subject"], df["run"] = subject, run
    return df


def list_runs() -> list[tuple[str, str]]:
    """[(S01, R01), (S01, R02), ...]"""
    files = sorted(DATA_DIR.glob("S*R*.txt"))
    if not files:
        raise FileNotFoundError(f"데이터가 없습니다: {DATA_DIR}\n"
                                "먼저 실행: .venv\\Scripts\\python.exe scripts\\download_data.py")
    return [(f.stem[:3], f.stem[3:]) for f in files]


def load_all() -> pd.DataFrame:
    return pd.concat([load_run(s, r) for s, r in list_runs()], ignore_index=True)
