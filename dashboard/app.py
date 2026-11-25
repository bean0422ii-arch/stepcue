"""14주차: StepCue 대시보드 (Streamlit). 보호자·연구자 화면.

MQTT `stepcue/event`(0.5초마다 동결 확률 등)와 `stepcue/action`(에이전트 행동)을 받아 1초마다 갱신한다.
원시 가속도는 받지 않는다(엣지 밖으로 나가지 않음).

화면
1. 실시간: 동결 확률 그래프, 메트로놈(큐) 상태, 보호자 알림, 걷기·걸음 속도, 결정 로그
2. "잘못된 큐였어요" 버튼 → logs/feedback.jsonl (LLM 에이전트가 오탐 상황을 찾는 재료)
   가상 사용자(데모용): 큐가 켜졌다 꺼질 때까지 정답 레이블에 동결이 한 번도 없으면 자동으로 누른다
3. LLM 에이전트의 승인 요청(승인/거절), 임계값 변경 기록표, 피드백 목록

실행: .venv\\Scripts\\streamlit.exe run dashboard\\app.py   (브로커 먼저 실행)
"""
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mqtt_common import HOST, PORT, TOPIC_ACTION, TOPIC_EVENT, connect  # noqa: E402
from records import (APPROVAL_PATH, CHANGES_PATH, CONFIG_PATH, FEEDBACK_PATH, apply_threshold,  # noqa: E402
                     read_json, read_jsonl, record_change, record_feedback, write_json)

CHART_SECONDS = 60          # 그래프에 보이는 최근 시간(데이터 시간 기준)
STALE_SECONDS = 3           # 이 시간(실제 시간) 동안 이벤트가 없으면 "재생 대기 중"
ACTION_TEXT = {"cue_start": "큐 시작", "cue_escalate": "큐 강화", "cue_stop": "큐 중지",
               "notify_caregiver": "보호자 알림", "no_action": "행동 안 함"}
STATE_TEXT = {"IDLE": "대기", "CUEING": "큐 재생 중", "ESCALATED": "큐 강화 중"}


class Hub:
    """MQTT 메시지를 받아 화면에 필요한 상태를 모아 두는 객체. Streamlit 프로세스에 하나만 만든다."""

    def __init__(self):
        self.lock = threading.Lock()
        self.virtual_user = False
        self.error = None
        self.clear()
        try:
            self.client = connect("dashboard", retries=3)
        except SystemExit as e:
            self.client, self.error = None, str(e)
            return
        self.client.on_message = self.on_message
        self.client.on_connect = lambda c, *a: c.subscribe([(TOPIC_EVENT, 0), (TOPIC_ACTION, 0)])  # 재연결 대비
        self.client.subscribe([(TOPIC_EVENT, 0), (TOPIC_ACTION, 0)])

    def clear(self):
        self.events = deque(maxlen=CHART_SECONDS * 2 * 2)
        self.actions = deque(maxlen=300)
        self.cue = None              # 마지막 cue_start 행동
        self.cue_active = False
        self.cue_saw_fog = False     # 이 큐가 켜진 동안 정답 레이블에 동결이 있었나(가상 사용자 판단용)
        self.prev_action = None      # 마지막 큐 바로 전의 행동
        self.last_real = None        # 마지막 행동(no_action 제외)
        self.alert = None
        self.fed_back = set()        # 이미 피드백한 큐(시작 시각)
        self.auto_count = 0
        self.last_wall = None

    def on_message(self, client, userdata, msg):
        d = json.loads(msg.payload)
        with self.lock:
            self.last_wall = time.time()
            if msg.topic == TOPIC_EVENT:
                if self.events and d["t"] < self.events[-1]["t"] - 1:     # 시간이 되돌아감 = 새 재생
                    vu = self.virtual_user
                    self.clear()
                    self.virtual_user = vu
                self.events.append(d)
                if self.cue_active and d.get("truth") == 2:
                    self.cue_saw_fog = True
                return
            self.actions.append(d)
            a = d["action"]
            if a == "cue_start":
                self.prev_action, self.cue, self.cue_active = self.last_real, d, True
                self.cue_saw_fog = d.get("truth") == 2
            elif a == "cue_stop":
                self.cue_active = False
                if self.virtual_user and self.cue and not self.cue_saw_fog:
                    self._feedback("가상 사용자", "큐가 켜진 동안 정답 레이블에 동결 없음")
                    self.auto_count += 1
            elif a == "notify_caregiver":
                self.alert = d
            if a != "no_action":
                self.last_real = d

    def _feedback(self, by: str, note: str | None = None) -> bool:
        """lock 안에서 호출. 같은 큐에는 한 번만 기록."""
        if self.cue is None or self.cue["t"] in self.fed_back:
            return False
        record_feedback(self.cue, by=by, source=self.cue.get("source"), prev_action=self.prev_action, note=note)
        self.fed_back.add(self.cue["t"])
        return True

    def snapshot(self) -> dict:
        with self.lock:
            return {"events": list(self.events), "actions": list(self.actions), "cue": self.cue,
                    "cue_active": self.cue_active, "alert": self.alert, "fed_back": set(self.fed_back),
                    "auto_count": self.auto_count, "last_wall": self.last_wall}


@st.cache_resource
def get_hub() -> Hub:
    return Hub()


# ---------- 화면 조각 ----------
def metronome(bpm: int, vibrate: bool) -> str:
    """큐 템포에 맞춰 깜빡이는 점 (실제 소리 대신 화면으로 표시)."""
    period = 60 / bpm
    vib = "  + 진동" if vibrate else ""
    return f"""
<style>@keyframes stepcue-beat {{0% {{opacity:1; transform:scale(1.25)}} 30% {{opacity:.25; transform:scale(1)}}
100% {{opacity:.25}}}}</style>
<div style="display:flex;align-items:center;gap:.6rem;font-size:1.3rem;font-weight:600">
  <span style="display:inline-block;width:1.1rem;height:1.1rem;border-radius:50%;background:#e8590c;
   animation:stepcue-beat {period:.3f}s infinite"></span>메트로놈 {bpm} bpm{vib}</div>"""


def prob_chart(ev: pd.DataFrame, acts: pd.DataFrame, thr: float) -> alt.LayerChart:
    t_max = ev["t"].max()
    ev = ev[ev["t"] >= t_max - CHART_SECONDS].copy()
    ev["정답 동결"] = (ev["truth"] == 2).astype(float)
    x = alt.X("t:Q", title="재생 시각 (초)", scale=alt.Scale(domain=[t_max - CHART_SECONDS, t_max]))
    truth = alt.Chart(ev).mark_area(color="#e03131", opacity=0.12, interpolate="step-after").encode(
        x=x, y=alt.Y("정답 동결:Q", scale=alt.Scale(domain=[0, 1])))
    line = alt.Chart(ev).mark_line(color="#1c7ed6", strokeWidth=2).encode(
        x=x, y=alt.Y("prob_avg:Q", title="동결 확률 (3윈도우 평균)", scale=alt.Scale(domain=[0, 1])),
        tooltip=["t", "prob_avg", "walking", "cadence"])
    rule = alt.Chart(pd.DataFrame({"thr": [thr]})).mark_rule(color="#e03131", strokeDash=[6, 4]).encode(y="thr:Q")
    layers = [truth, line, rule]
    if len(acts):
        a = acts[(acts["t"].notna()) & (acts["t"] >= t_max - CHART_SECONDS) & (acts["action"] != "no_action")].copy()
        if len(a):
            a["행동"] = a["action"].map(ACTION_TEXT)
            layers.append(alt.Chart(a).mark_point(filled=True, size=90, color="#e8590c").encode(
                x="t:Q", y=alt.Y("prob_avg:Q"), shape="행동:N", tooltip=["t", "행동", "reason"]))
    return alt.layer(*layers).properties(height=280)


# ---------- 페이지 ----------
st.set_page_config(page_title="StepCue", layout="wide")
hub = get_hub()
st.title("StepCue 대시보드")
if hub.error:
    st.error(f"MQTT 브로커({HOST}:{PORT})에 연결하지 못했습니다. 브로커를 먼저 실행한 뒤 이 페이지를 새로고침하세요.\n\n"
             "`.venv\\Scripts\\python.exe src\\broker.py`")
    get_hub.clear()
    st.stop()

with st.sidebar:
    cfg = read_json(CONFIG_PATH) or {}
    st.markdown(f"**브로커** {HOST}:{PORT}  \n**현재 임계값** {cfg.get('threshold')}")
    hub.virtual_user = st.toggle("가상 사용자 (데모)", value=hub.virtual_user,
                                 help="큐가 켜졌다 꺼질 때까지 정답 레이블에 동결이 없었으면 '잘못된 큐'를 자동으로 누릅니다.")
    st.caption("빨간 배경은 데이터셋의 정답 레이블(동결)입니다. 평가용으로만 표시하고 에이전트 판단에는 쓰지 않습니다.")


@st.fragment(run_every=1.0)
def live():
    s = hub.snapshot()
    thr = (read_json(CONFIG_PATH) or {}).get("threshold", 0.8)
    if not s["events"]:
        st.info("데이터를 기다리는 중입니다. 엣지 노드와 시뮬레이터(--mqtt)를 실행하세요.")
        return
    ev = pd.DataFrame(s["events"])
    acts = pd.DataFrame(s["actions"])
    last = s["events"][-1]
    stale = time.time() - s["last_wall"] > STALE_SECONDS
    state = (s["actions"][-1].get("state", "IDLE") if s["actions"] else "IDLE") if s["cue_active"] else "IDLE"

    if s["alert"] and last["t"] - (s["alert"]["t"] or last["t"]) < 30:
        st.error(f"보호자 알림 (t={s['alert']['t']:.1f}s): {s['alert']['reason']}")
    if stale:
        st.warning("최근 몇 초 동안 데이터가 들어오지 않았습니다(재생 끝 또는 센서 끊김).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("동결 확률", f"{last['prob_avg']:.2f}", f"임계값 {thr}", delta_color="off")
    c2.metric("에이전트 상태", STATE_TEXT.get(state, state))
    c3.metric("걷기", "걷는 중" if last["walking"] else "걷지 않음", f"리듬 {last['rhythm']}", delta_color="off")
    c4.metric("걸음 속도", f"{last['cadence']} 걸음/분" if last["cadence"] else "-")

    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(prob_chart(ev, acts, thr), width="stretch")
    with right:
        cue = s["cue"]
        if s["cue_active"]:
            cur = next((a for a in reversed(s["actions"]) if a.get("bpm")), cue)
            st.markdown(metronome(cur["bpm"], bool(cur.get("vibrate"))), unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:1.3rem;opacity:.6'>메트로놈 꺼짐</div>", unsafe_allow_html=True)
        can_feedback = cue is not None and cue["t"] not in s["fed_back"] and last["t"] - cue["t"] < 60
        label = f"잘못된 큐였어요 (t={cue['t']:.1f}s 큐)" if cue else "잘못된 큐였어요"
        if st.button(label, disabled=not can_feedback, type="primary"):
            with hub.lock:
                ok = hub._feedback("사용자")
            st.toast("피드백을 기록했습니다." if ok else "이미 기록된 큐입니다.")
        if hub.virtual_user:
            st.caption(f"가상 사용자 자동 피드백 {s['auto_count']}회")
        st.markdown("**결정 로그** (최근)")
        if len(acts):
            show = acts.tail(12).iloc[::-1]
            st.dataframe(pd.DataFrame({"t": show["t"], "행동": show["action"].map(ACTION_TEXT),
                                       "bpm": show.get("bpm"), "이유": show["reason"]}),
                         hide_index=True, width="stretch", height=260)
        st.caption(f"재생 t={last['t']:.1f}s · 엣지 판단 지연 {last.get('latency_ms')} ms")


@st.fragment(run_every=3.0)
def records_panel():
    st.subheader("LLM 에이전트 제안")
    req = read_json(APPROVAL_PATH)
    if req and req.get("status") == "pending":
        st.warning(f"**임계값 {req['from']} → {req['to']}** 변경 승인 요청\n\n이유: {req['reason']}"
                   + (f"\n\n예상 효과: {req['expected']}" if req.get("expected") else ""))
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("승인"):
            apply_threshold(req["to"], req["reason"], "사람 승인", req.get("expected"))
            write_json(APPROVAL_PATH, {**req, "status": "approved", "decided": round(time.time(), 3)})
            st.rerun(scope="fragment")
        if b2.button("거절"):
            record_change(req["from"], req["to"], req["reason"], "거절", False, req.get("expected"))
            write_json(APPROVAL_PATH, {**req, "status": "rejected", "decided": round(time.time(), 3)})
            st.rerun(scope="fragment")
    else:
        st.caption("승인을 기다리는 요청이 없습니다.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**임계값 변경 기록표**")
        ch = read_jsonl(CHANGES_PATH)
        if ch:
            d = pd.DataFrame(ch)
            st.dataframe(pd.DataFrame({
                "시각": pd.to_datetime(d["wall"], unit="s", utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%m-%d %H:%M"),
                "변경": d["from"].astype(str) + " → " + d["to"].astype(str), "이유": d["reason"],
                "예상 효과": d["expected"].astype(str), "승인": d["approval"], "적용 후 효과": d["after"].astype(str)}),
                hide_index=True, width="stretch")
        else:
            st.caption("아직 없음")
    with col2:
        fb = read_jsonl(FEEDBACK_PATH)
        st.markdown(f"**\"잘못된 큐\" 피드백** ({len(fb)}건)")
        if fb:
            d = pd.DataFrame(fb).tail(10).iloc[::-1]
            st.dataframe(d[["by", "source", "t_cue", "prob_avg", "walking", "rhythm", "cadence", "prev_action"]]
                         .rename(columns={"by": "누가", "source": "재생", "t_cue": "큐 시각", "prob_avg": "확률",
                                          "walking": "걷기", "rhythm": "리듬", "cadence": "걸음/분",
                                          "prev_action": "직전 행동"}),
                         hide_index=True, width="stretch")
        else:
            st.caption("아직 없음")


live()
st.divider()
records_panel()
