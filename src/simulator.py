"""12주차: 웨어러블 시뮬레이터. 녹화된 Daphnet 데이터를 64Hz 실시간처럼 재생한다.

13주차에는 replay()가 내보내는 샘플을 MQTT `stepcue/imu`로 발행한다. 지금은 콘솔 출력만.

실행 예시
  .venv\\Scripts\\python.exe src\\simulator.py --subject S01 --run R01 --start 60 --duration 30
  .venv\\Scripts\\python.exe src\\simulator.py --subject S01 --run R01 --start 60 --duration 30 --infer
  (--speed 4 : 4배속, --infer : 2초 윈도우마다 학습한 모델로 동결 확률 출력. 13주차 엣지 노드 미리보기)
"""
import argparse
import pickle
import time
from collections import deque
from typing import Iterator

import numpy as np

from data import ACC_COLS, FS, ROOT, load_run
from features import STEP, WIN, extract


def replay(subject: str, run: str, start: float = 0.0, duration: float | None = None,
           speed: float = 1.0) -> Iterator[dict]:
    """샘플을 하나씩, 실제 시간 간격(1/64초 ÷ speed)에 맞춰 내보낸다.

    label은 정답 확인(시나리오 평가)용이고, 엣지 노드의 판단에는 쓰지 않는다.
    """
    df = load_run(subject, run)
    t = (df["time_ms"].to_numpy() - df["time_ms"].iloc[0]) / 1000
    sel = (t >= start) & (t < start + duration if duration else True)
    acc = df.loc[sel, ACC_COLS].to_numpy(dtype=float)
    labels = df.loc[sel, "label"].to_numpy()
    times = t[sel]

    t0 = time.perf_counter()
    for i in range(len(acc)):
        wait = t0 + i / (FS * speed) - time.perf_counter()     # 밀림이 쌓이지 않게 시작 시각 기준으로 맞춤
        if wait > 0:
            time.sleep(wait)
        yield {"t": round(float(times[i]), 4), "acc": acc[i].tolist(), "label": int(labels[i])}


LABEL_TEXT = {0: "실험외", 1: "정상", 2: "동결"}


def main():
    ap = argparse.ArgumentParser(description="Daphnet 실시간 재생")
    ap.add_argument("--subject", default="S01")
    ap.add_argument("--run", default="R01")
    ap.add_argument("--start", type=float, default=0.0, help="시작 시각(초)")
    ap.add_argument("--duration", type=float, default=30.0, help="재생 길이(초), 0이면 끝까지")
    ap.add_argument("--speed", type=float, default=1.0, help="재생 배속")
    ap.add_argument("--infer", action="store_true", help="모델로 동결 확률 계산(models/rf.pkl 필요)")
    a = ap.parse_args()

    clf = None
    if a.infer:
        with open(ROOT / "models" / "rf.pkl", "rb") as f:
            clf = pickle.load(f)
        clf.n_jobs = 1          # 윈도우 1개씩 예측할 때는 병렬 처리 준비 비용이 더 커서 1코어가 빠름
    buf = deque(maxlen=WIN)

    print(f"재생: {a.subject}{a.run}, {a.start}초부터 {a.duration or '끝'}초, {a.speed}배속")
    n, t_wall = 0, time.perf_counter()
    for n, s in enumerate(replay(a.subject, a.run, a.start, a.duration or None, a.speed), start=1):
        buf.append(s["acc"])
        if n % STEP:                                   # 0.5초(32샘플)마다 한 줄 출력
            continue
        ankle = np.linalg.norm(s["acc"][:3])
        line = f"t={s['t']:7.2f}s  발목|a|={ankle:6.0f}mg  정답={LABEL_TEXT[s['label']]}"
        if clf is not None and len(buf) == WIN:
            t1 = time.perf_counter()
            prob = clf.predict_proba(extract(np.array(buf)))[0, 1]
            ms = (time.perf_counter() - t1) * 1000
            line += f"  동결확률={prob:.2f} {'#' * int(prob * 20):<20} ({ms:.1f}ms)"
        print(line, flush=True)

    elapsed = time.perf_counter() - t_wall
    print(f"샘플 {n}개, 실제 걸린 시간 {elapsed:.1f}초 (기대 {n / FS / a.speed:.1f}초)")


if __name__ == "__main__":
    main()
