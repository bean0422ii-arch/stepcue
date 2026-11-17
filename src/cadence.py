"""12주차: 개인 걸음 속도(cadence, 분당 걸음 수) 추정. 에이전트의 큐 기본 템포로 쓴다.

방법 (발목 가속도, 최근 10초)
1. 3축 크기 |a|를 0.5~3Hz로 필터링(걷기 대역만 남김)
2. 자기상관: 신호를 시간만큼 밀어서 원래 신호와 얼마나 닮았는지 계산
3. 0.8~1.6초 사이에서 가장 닮은 시간 = 한 보폭(왼발 + 오른발) 시간
4. 분당 걸음 수 = 2걸음 × 60초 / 보폭 시간
   닮은 정도가 낮으면(리듬 있는 걷기가 아니면) None -> 템포를 갱신하지 않음

처음에는 "피크 간격"으로 계획했지만, 발목 센서가 한쪽 다리에만 있어서 피크가 한 걸음마다 잡히기도 하고
한 보폭마다 잡히기도 했다. 자기상관은 보폭 주기에서 항상 가장 크게 나와서 이 방법으로 바꿨다.

실행: .venv\\Scripts\\python.exe src\\cadence.py   -> docs/cadence.md (환자별 걸음 속도 표)
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from data import FS, ROOT, list_runs, load_run

SEG = 10 * FS                  # 10초 구간
STRIDE_RANGE = (0.8, 1.6)      # 보폭 시간 범위(초) = 분당 75~150걸음
MIN_RHYTHM = 0.4               # 자기상관 최소값: 이보다 낮으면 걷는 중이 아니라고 판단
MIN_STD = 50                   # 필터 후 표준편차(mg): 거의 움직이지 않으면 계산하지 않음
_SOS = butter(2, [0.5, 3.0], btype="band", fs=FS, output="sos")


def rhythm(ankle_xyz: np.ndarray) -> tuple[float, float]:
    """(보폭 시간 초, 자기상관 값). ankle_xyz: (n, 3), 10초 정도 권장."""
    m = np.linalg.norm(np.asarray(ankle_xyz, dtype=float), axis=1)
    x = sosfiltfilt(_SOS, m - m.mean())
    if x.std() < MIN_STD:
        return np.nan, 0.0
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    lo, hi = int(STRIDE_RANGE[0] * FS), int(STRIDE_RANGE[1] * FS)
    lag = lo + int(np.argmax(ac[lo:hi + 1]))
    return lag / FS, float(ac[lag])


def estimate_cadence(ankle_xyz: np.ndarray) -> float | None:
    """분당 걸음 수. 리듬 있는 걷기가 아니면 None."""
    stride, r = rhythm(ankle_xyz)
    if r < MIN_RHYTHM:
        return None
    return 2 * 60 / stride


def main():
    rows = []
    for s, r in list_runs():
        df = load_run(s, r)
        acc = df[["ankle_x", "ankle_y", "ankle_z"]].to_numpy(float)
        lab = df["label"].to_numpy()
        for st in range(0, len(df) - SEG + 1, SEG // 2):
            if (lab[st:st + SEG] == 1).all():              # 정상(동결 아님) 구간만
                rows.append({"subject": s, "spm": estimate_cadence(acc[st:st + SEG])})
    d = pd.DataFrame(rows)
    table = d.groupby("subject")["spm"].agg(
        구간수="size",
        걷기로판단="count",
        중앙값=lambda v: v.median(),
        하위25=lambda v: v.quantile(0.25),
        상위75=lambda v: v.quantile(0.75)).round(1)
    table = table.rename(columns={"구간수": "10초 구간 수", "걷기로판단": "걷기로 판단된 구간",
                                  "중앙값": "걸음/분 중앙값", "하위25": "하위 25%", "상위75": "상위 75%"})
    print(table.to_string())
    md = ["# 환자별 걸음 속도 (12주차)", "",
          "`src/cadence.py`로 생성. 정상(레이블 1) 10초 구간(5초 간격)마다 발목 가속도 자기상관으로 보폭 주기를 찾아 "
          "분당 걸음 수로 바꿨다. 서 있거나 방향 전환 중이라 리듬이 약한 구간(자기상관 < "
          f"{MIN_RHYTHM})은 제외.", "",
          table.to_markdown(), "",
          f"- 전체 중앙값: {d['spm'].median():.0f} 걸음/분, 환자별 중앙값 범위: "
          f"{table['걸음/분 중앙값'].min():.0f} ~ {table['걸음/분 중앙값'].max():.0f} 걸음/분",
          "- 환자마다 걸음 속도가 달라서, 고정 템포(예: 100 bpm) 대신 개인 걸음 속도를 큐 템포로 쓰는 근거가 된다.",
          "- 에이전트는 실행 중 최근 10초로 같은 계산을 반복해 템포를 갱신한다(이동 평균).", ""]
    (ROOT / "docs" / "cadence.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
