"""13주차: 엣지 노드. `stepcue/imu`를 구독 -> 2초 윈도우 -> 특징 추출 -> 모델 추론 -> `stepcue/event` 발행.

원시 가속도는 엣지 안에서만 쓰고, 밖으로는 0.5초마다 요약(확률, 걷기 여부, 걸음 속도)만 내보낸다.
임계값은 config/user.json에서 읽고, 파일이 바뀌면 다시 읽는다(14주차 LLM 에이전트가 바꿀 수 있게).

실행: .venv\\Scripts\\python.exe src\\edge_node.py   (브로커 먼저 실행, Ctrl+C로 종료)

발행 예시 (stepcue/event)
  {"t": 1140.5, "fog_prob": 0.91, "prob_avg": 0.86, "fog": true, "threshold": 0.8,
   "walking": false, "rhythm": 0.21, "cadence": null, "infer_ms": 1.4, "latency_ms": 2.1, "truth": 2}
"""
import json
import pickle
import time
from collections import deque

import numpy as np

from cadence import MIN_RHYTHM, SEG, rhythm
from data import FS, ROOT
from features import STEP, WIN, extract
from mqtt_common import TOPIC_EVENT, TOPIC_IMU, connect

MODEL_PATH = ROOT / "models" / "rf.pkl"
CONFIG_PATH = ROOT / "config" / "user.json"


class EdgeNode:
    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self.clf = pickle.load(f)
        self.clf.n_jobs = 1                    # 윈도우 1개씩 예측할 때는 1코어가 빠름 (19ms -> 1.5ms)
        self.cfg, self.cfg_mtime = {}, 0.0
        self.reload_config()
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=SEG)           # 최근 10초 (걸음 리듬 계산용). 마지막 2초가 모델 입력
        self.probs = deque(maxlen=self.cfg["smooth_windows"])
        self.n = 0

    def reload_config(self):
        mtime = CONFIG_PATH.stat().st_mtime
        if mtime != self.cfg_mtime:
            self.cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if self.cfg_mtime:
                print(f"설정 변경 감지 -> 임계값 {self.cfg['threshold']}", flush=True)
            self.cfg_mtime = mtime

    def on_sample(self, s: dict) -> dict | None:
        """샘플 1개를 받는다. 0.5초(32샘플)마다, 2초가 찼으면 이벤트를 돌려준다."""
        self.buf.append(s["acc"])
        self.n += 1
        if self.n % STEP or len(self.buf) < WIN:
            return None

        self.reload_config()
        t1 = time.perf_counter()
        acc = np.asarray(self.buf)
        prob = float(self.clf.predict_proba(extract(acc[-WIN:]))[0, 1])
        self.probs.append(prob)
        avg = float(np.mean(self.probs))
        stride, r = rhythm(acc[:, :3]) if len(acc) >= 4 * FS else (np.nan, 0.0)   # 4초 이상 쌓인 뒤 계산
        walking = r >= MIN_RHYTHM
        infer_ms = (time.perf_counter() - t1) * 1000

        return {"t": s["t"], "fog_prob": round(prob, 3), "prob_avg": round(avg, 3),
                "fog": avg >= self.cfg["threshold"], "threshold": self.cfg["threshold"],
                "walking": bool(walking), "rhythm": round(r, 2),
                "cadence": round(2 * 60 / stride) if walking else None,
                "infer_ms": round(infer_ms, 2),
                "latency_ms": round((time.time() - s["sent"]) * 1000, 1) if "sent" in s else None,
                "truth": s.get("label")}      # 정답 레이블: 평가·대시보드 표시용, 판단에는 안 씀


LABEL_TEXT = {0: "실험외", 1: "정상", 2: "동결", None: "-"}


def main():
    node = EdgeNode()
    client = connect("edge")

    def on_message(cl, userdata, msg):
        s = json.loads(msg.payload)
        if s.get("reset"):
            node.reset()
            print(f"--- 새 재생: {s.get('source')} ---", flush=True)
            return
        ev = node.on_sample(s)
        if ev is None:
            return
        cl.publish(TOPIC_EVENT, json.dumps(ev))
        bar = "#" * int(ev["prob_avg"] * 20)
        print(f"t={ev['t']:7.2f}s  확률(평균)={ev['prob_avg']:.2f} {bar:<20} "
              f"{'동결!' if ev['fog'] else '     '}  걷기={'O' if ev['walking'] else 'X'} "
              f"걸음={ev['cadence'] or '-':>3}  추론 {ev['infer_ms']:.1f}ms  지연 {ev['latency_ms']}ms  "
              f"정답={LABEL_TEXT[ev['truth']]}", flush=True)

    client.on_message = on_message
    client.subscribe(TOPIC_IMU)
    print(f"엣지 노드 실행 중: {TOPIC_IMU} 구독, 임계값 {node.cfg['threshold']}  (Ctrl+C로 종료)", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("엣지 노드 종료")
    finally:
        client.disconnect()
        client.loop_stop()


if __name__ == "__main__":
    main()
