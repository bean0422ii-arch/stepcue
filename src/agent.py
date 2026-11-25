"""14주차: 빠른 층 에이전트. 엣지 이벤트(0.5초마다)를 받아 큐를 켜고, 강화하고, 끄고, 보호자에게 알린다.

규칙 기반 상태 머신이라 1초 안에, 네트워크 없이 동작한다(LLM 없음). 엣지 노드 프로세스 안에서 직접 호출되므로
MQTT 브로커가 끊겨도 판단과 큐는 계속된다. 대시보드로는 행동만 `stepcue/action`으로 발행한다.

상태
  IDLE ──확률 ≥ 임계값──> CUEING ──3초 미회복──> ESCALATED
    ^                        │                      │
    └──확률 < 임계값 2윈도우 연속(회복)──┴──────────────────────┘
  큐 시작 후 10초 넘게 미회복 → 보호자 알림 (큐 1회당 최대 1번)
  10분 안에 큐 3회째 → 보호자 알림 (한 번 알리면 10분 동안 같은 이유로 다시 알리지 않음)

"직전 5초 걷는 중" 조건(require_walking)은 시험했지만 기본으로 쓰지 않는다(docs/agent_results.md).
실제 동결 직전에 걷고 있던 비율(58%)과 오탐 직전 비율(60%)이 거의 같아서, 조건을 켜면 오탐은 조금 줄지만
(10분당 4.5 → 3.4회) 감지율이 80.6% → 49.8%로 무너졌다. 비교 실험용 옵션으로만 남긴다.
걷기 여부는 계속 기록해서 LLM 에이전트가 "잘못된 큐" 피드백의 상황을 분석할 때 쓴다.

기록(logs/decisions.jsonl): 모든 행동과, 확률이 임계값 근처(임계값 − 0.2 이상)까지 올랐지만 행동하지 않은
판단(no_action)을 입력값·이유와 함께 남긴다. 확률이 그보다 낮은 0.5초마다의 판단은 너무 많아서 개수만 센다(summary()).

단독 확인: .venv\\Scripts\\python.exe src\\agent.py   (가짜 이벤트로 상태 변화를 출력)
"""
import json
import time
from collections import deque
from pathlib import Path

from data import ROOT

DECISIONS_PATH = ROOT / "logs" / "decisions.jsonl"

IDLE, CUEING, ESCALATED = "IDLE", "CUEING", "ESCALATED"
WALK_LOOKBACK = 5.0        # 초: 이 시간 안에 걷고 있었어야 큐를 시작
RECOVER_WINDOWS = 2        # 확률이 임계값 아래로 이만큼 연속이면 회복
ESCALATE_AFTER = 3.0       # 초: 큐 시작 후 이 시간까지 회복이 없으면 강화
ESCALATE_TEMPO = 0.9       # 강화 시 템포 = 기본 템포 × 0.9
NOTIFY_AFTER = 10.0        # 초: 큐 시작 후 이 시간까지 회복이 없으면 보호자 알림
NOTIFY_COUNT, NOTIFY_SPAN = 3, 600.0   # 10분 안에 큐 3회째면 보호자 알림
DEFAULT_BPM = 100          # 걸음 속도를 아직 모를 때의 템포 (환자별 중앙값 95~130, docs/cadence.md)
CADENCE_HISTORY = 10       # 기본 템포 = 최근 걷기 이벤트 10개의 걸음 속도 중앙값
NEAR_MARGIN = 0.2          # 확률이 (임계값 − 이 값) 이상이면 "행동 안 함"도 이유와 함께 기록
SENSOR_TIMEOUT = 2.0       # 초(실제 시간): 이벤트가 이만큼 안 오면 센서 끊김으로 보고 큐 중지


class Agent:
    def __init__(self, log_path: Path | None = DECISIONS_PATH, require_walking: bool = False):
        """log_path=None이면 파일에 기록하지 않음(오프라인 평가용).
        require_walking=True면 "직전 5초 걷는 중" 조건을 켠다(효과 비교 실험용, 기본은 끔)."""
        self.log_path = log_path
        self.require_walking = require_walking
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self, source: str | None = None):
        """새 재생(세션) 시작: 상태와 기록을 비운다."""
        self.source = source
        self.state = IDLE
        self.last_walk_t = None        # 마지막으로 걷기가 감지된 시각(데이터 시간)
        self.cadences = deque(maxlen=CADENCE_HISTORY)
        self.cue_t = None              # 현재 큐 시작 시각
        self.bpm = None
        self.below = 0                 # 임계값 아래 연속 윈도우 수
        self.notified = False          # 현재 큐에서 보호자 알림을 이미 보냈는지
        self.cue_starts = deque()      # 최근 10분 큐 시작 시각
        self.freq_notify_t = None      # "잦은 동결" 알림을 마지막으로 보낸 시각
        self.last_event_wall = None    # 마지막 이벤트를 받은 실제 시각(센서 끊김 감지용)
        self.counts = {"events": 0, "idle_low": 0}

    # ---------- 도구 ----------
    def base_bpm(self) -> int:
        """기본 템포 = 최근 걷기 중 걸음 속도의 중앙값 (모르면 100)."""
        if not self.cadences:
            return DEFAULT_BPM
        c = sorted(self.cadences)
        return int(round(c[len(c) // 2]))

    def _act(self, ev: dict, action: str, reason: str, **extra) -> dict:
        """행동 하나를 만들고 기록한다. 대시보드로 보낼 메시지도 이것."""
        rec = {"t": ev["t"], "action": action, **extra, "reason": reason, "state": self.state,
               "prob_avg": ev["prob_avg"], "threshold": ev["threshold"], "walking": ev["walking"],
               "rhythm": ev.get("rhythm"), "cadence": ev.get("cadence"),
               "truth": ev.get("truth")}           # 정답은 기록(평가)용, 판단에는 쓰지 않음
        self.log_decision(rec)
        return rec

    def log_decision(self, rec: dict):
        if self.log_path is None:
            return
        line = {"wall": round(time.time(), 3), "source": self.source, **rec}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # ---------- 판단 ----------
    def step(self, ev: dict) -> list[dict]:
        """엣지 이벤트 하나를 받아 이번에 할 행동 목록을 돌려준다(대부분 빈 목록)."""
        self.counts["events"] += 1
        self.last_event_wall = time.time()
        t, high = ev["t"], ev["prob_avg"] >= ev["threshold"]
        if ev["walking"]:
            self.last_walk_t = t
            if ev.get("cadence") and self.state == IDLE:     # 큐가 켜진 동안의 걸음은 템포 추정에서 뺌
                self.cadences.append(ev["cadence"])
        walked = self.last_walk_t is not None and t - self.last_walk_t <= WALK_LOOKBACK

        if self.state == IDLE:
            if not high:
                if ev["prob_avg"] >= ev["threshold"] - NEAR_MARGIN:
                    return [self._act(ev, "no_action", f"확률 {ev['prob_avg']:.2f}로 올랐지만 임계값 "
                                      f"{ev['threshold']} 미만 → 동결로 보지 않고 행동 안 함")]
                self.counts["idle_low"] += 1
                return []
            if self.require_walking and not walked:
                return [self._act(ev, "no_action", f"확률 {ev['prob_avg']:.2f}로 높지만 직전 "
                                  f"{WALK_LOOKBACK:.0f}초 동안 걷지 않음 → 일부러 멈춘 것으로 판단")]
            return self._start_cue(ev, t)

        # CUEING 또는 ESCALATED
        self.below = 0 if high else self.below + 1
        if self.below >= RECOVER_WINDOWS:
            dur = t - self.cue_t
            self.state = IDLE
            return [self._act(ev, "cue_stop", f"확률이 임계값 아래로 {RECOVER_WINDOWS}윈도우 연속 → 회복 "
                              f"(큐 {dur:.1f}초)")]
        acts, dur = [], t - self.cue_t
        if self.state == CUEING and dur > ESCALATE_AFTER:
            self.state = ESCALATED
            self.bpm = int(round(self.bpm * ESCALATE_TEMPO))
            acts.append(self._act(ev, "cue_escalate", f"큐 시작 후 {dur:.1f}초 동안 회복 안 됨 → "
                                  f"템포 {ESCALATE_TEMPO:.0%}로 낮추고 진동 추가", bpm=self.bpm, vibrate=True))
        if dur > NOTIFY_AFTER and not self.notified:
            self.notified = True
            acts.append(self._act(ev, "notify_caregiver",
                                  f"동결이 {dur:.0f}초째 계속됨 → 낙상 위험, 보호자 알림"))
        return acts

    def _start_cue(self, ev: dict, t: float) -> list[dict]:
        self.state, self.cue_t, self.below, self.notified = CUEING, t, 0, False
        self.bpm = self.base_bpm()
        known = "개인 걸음 속도" if self.cadences else "기본값(걸음 속도 아직 모름)"
        acts = [self._act(ev, "cue_start", f"동결 확률 {ev['prob_avg']:.2f} ≥ {ev['threshold']} "
                          f"→ 메트로놈 {self.bpm}bpm ({known})", bpm=self.bpm, vibrate=False)]
        self.cue_starts.append(t)
        while self.cue_starts and t - self.cue_starts[0] > NOTIFY_SPAN:
            self.cue_starts.popleft()
        quiet = self.freq_notify_t is None or t - self.freq_notify_t > NOTIFY_SPAN
        if len(self.cue_starts) >= NOTIFY_COUNT and quiet:
            self.notified = True
            self.freq_notify_t = t
            acts.append(self._act(ev, "notify_caregiver", f"최근 {NOTIFY_SPAN / 60:.0f}분 안에 "
                                  f"큐 {len(self.cue_starts)}회 → 동결이 잦음, 보호자 알림"))
        return acts

    def check_sensor(self, now: float | None = None) -> list[dict]:
        """주기적으로 호출. 큐가 켜진 채로 이벤트가 끊기면 큐를 끈다(근거 없는 큐 방지)."""
        now = time.time() if now is None else now
        if self.state == IDLE or self.last_event_wall is None or now - self.last_event_wall < SENSOR_TIMEOUT:
            return []
        self.state = IDLE
        rec = {"t": None, "action": "cue_stop", "reason": f"센서 데이터가 {now - self.last_event_wall:.1f}초 "
               "동안 안 들어옴 → 근거 없는 큐를 막기 위해 중지", "state": self.state}
        self.log_decision(rec)
        return [rec]

    def summary(self) -> dict:
        return {"state": self.state, "base_bpm": self.base_bpm(), **self.counts}


def _demo():
    """가짜 이벤트로 상태 변화 확인: 걷기 → 동결 13초 → 회복, 그리고 서 있다가 확률이 오르는 경우.
    마지막 경우는 걷기 조건을 켠 에이전트로만 no_action이 나온다(비교용)."""
    agent = Agent(log_path=None, require_walking=True)
    evs = []
    for i in range(20):                    # 10초 걷기 (걸음 속도 110)
        evs.append({"t": i * 0.5, "prob_avg": 0.2, "walking": True, "cadence": 110})
    for i in range(20, 46):                # 13초 동결 (걷기 리듬 사라짐)
        evs.append({"t": i * 0.5, "prob_avg": 0.9, "walking": False, "cadence": None})
    for i in range(46, 50):                # 회복
        evs.append({"t": i * 0.5, "prob_avg": 0.3, "walking": True, "cadence": 108})
    for i in range(50, 70):                # 10초 서 있음
        evs.append({"t": i * 0.5, "prob_avg": 0.2, "walking": False, "cadence": None})
    evs.append({"t": 35.0, "prob_avg": 0.85, "walking": False, "cadence": None})   # 서 있는데 확률 상승
    for ev in evs:
        for a in agent.step({"threshold": 0.8, **ev}):
            print(f"t={a['t']:5.1f}s  {a['action']:<17} {a.get('bpm') or '':>4}  {a['reason']}")
    print(agent.summary())


if __name__ == "__main__":
    _demo()
