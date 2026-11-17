"""12주차: 윈도우 분할과 특징 추출.

- 윈도우: 2초(128샘플), 0.5초(32샘플)마다 이동
- 특징 10개 (센서 방향에 영향받지 않도록 3축 가속도의 크기 |a| 사용)
    fi_ankle, fi_thigh, fi_trunk        Freeze Index = log(동결 대역 3~8Hz 파워 / 걷기 대역 0.5~3Hz 파워)
    pow_ankle, pow_thigh, pow_trunk     log(0.5~8Hz 전체 파워)  -> 서 있는지(움직임이 적은지) 판단
    loco_ankle, freeze_ankle            발목의 걷기 대역, 동결 대역 파워(log) 각각
    domf_ankle, domf_trunk              0.5~8Hz에서 가장 센 주파수(Hz)

Freeze Index 아이디어: 동결 중에는 다리가 3~8Hz로 떨리고(trembling), 정상 걷기는 0.5~3Hz에 에너지가 몰린다.
(Moore et al. 2008, Bächlin et al. 2010)

실행: .venv\\Scripts\\python.exe src\\features.py   -> data/processed/windows.csv.gz
"""
import numpy as np
import pandas as pd

from data import FS, ROOT, SENSORS, list_runs, load_run

WIN = 2 * FS        # 128 샘플
STEP = FS // 2      # 32 샘플
LOCO_BAND = (0.5, 3.0)
FREEZE_BAND = (3.0, 8.0)
EPS = 1e-6

FEATURE_NAMES = ([f"fi_{s}" for s in SENSORS] + [f"pow_{s}" for s in SENSORS]
                 + ["loco_ankle", "freeze_ankle", "domf_ankle", "domf_trunk"])
OUT_PATH = ROOT / "data" / "processed" / "windows.csv.gz"

_FREQS = np.fft.rfftfreq(WIN, d=1 / FS)            # 0, 0.5, 1.0, ... 32Hz
_HANN = np.hanning(WIN)
_LOCO = (_FREQS >= LOCO_BAND[0]) & (_FREQS < LOCO_BAND[1])
_FRZ = (_FREQS >= FREEZE_BAND[0]) & (_FREQS < FREEZE_BAND[1])
_ALL = (_FREQS >= LOCO_BAND[0]) & (_FREQS < FREEZE_BAND[1])


def extract(windows: np.ndarray) -> np.ndarray:
    """windows: (n, 128, 9) 가속도 [발목xyz, 허벅지xyz, 몸통xyz] -> (n, 10) 특징.

    엣지 노드에서는 n=1로 호출한다(윈도우 하나씩).
    """
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim == 2:
        windows = windows[None]
    n = windows.shape[0]
    mag = np.linalg.norm(windows.reshape(n, WIN, 3, 3), axis=3)    # (n, 128, 3센서)
    mag = mag - mag.mean(axis=1, keepdims=True)                     # 중력(평균) 제거
    spec = np.abs(np.fft.rfft(mag * _HANN[None, :, None], axis=1)) ** 2   # (n, 65, 3)

    loco = spec[:, _LOCO].sum(axis=1)       # (n, 3)
    frz = spec[:, _FRZ].sum(axis=1)
    total = spec[:, _ALL].sum(axis=1)
    fi = np.log((frz + EPS) / (loco + EPS))
    power = np.log(total + EPS)
    domf = _FREQS[_ALL][spec[:, _ALL].argmax(axis=1)]                # (n, 3)

    return np.column_stack([fi, power,
                            np.log(loco[:, 0] + EPS), np.log(frz[:, 0] + EPS),
                            domf[:, 0], domf[:, 2]])


def window_run(df: pd.DataFrame) -> pd.DataFrame:
    """한 run을 윈도우로 자르고 특징과 정답을 붙인다."""
    acc = df[[c for c in df.columns if c.split("_")[0] in SENSORS]].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy()
    starts = np.arange(0, len(df) - WIN + 1, STEP)
    idx = starts[:, None] + np.arange(WIN)[None, :]
    feats = extract(acc[idx])

    lab = labels[idx]
    out = pd.DataFrame(feats, columns=FEATURE_NAMES)
    out.insert(0, "subject", df["subject"].iloc[0])
    out.insert(1, "run", df["run"].iloc[0])
    out.insert(2, "t_end", (starts + WIN - 1) / FS)       # 윈도우 끝 시각(초), 실시간이면 이때 판단 가능
    out["has_exp0"] = (lab == 0).any(axis=1)              # 실험 외 구간이 섞인 윈도우
    out["frac_fog"] = (lab == 2).mean(axis=1)
    out["y"] = (out["frac_fog"] >= 0.5).astype(int)       # 다수결: 절반 이상 동결이면 1
    return out


def build() -> pd.DataFrame:
    frames = [window_run(load_run(s, r)) for s, r in list_runs()]
    return pd.concat(frames, ignore_index=True)


def load_windows() -> pd.DataFrame:
    """학습·평가용 윈도우(실험 외 구간이 섞인 윈도우는 제외)."""
    if not OUT_PATH.exists():
        main()
    w = pd.read_csv(OUT_PATH)
    return w[~w["has_exp0"]].reset_index(drop=True)


def main():
    w = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(OUT_PATH, index=False, compression="gzip")
    used = w[~w["has_exp0"]]
    print(f"윈도우 {len(w)}개 (학습·평가 사용 {len(used)}개, 동결 {used['y'].sum()}개) -> {OUT_PATH}")
    print(used.groupby("y")[FEATURE_NAMES].median().T.round(2).rename(columns={0: "정상 중앙값", 1: "동결 중앙값"}))


if __name__ == "__main__":
    main()
