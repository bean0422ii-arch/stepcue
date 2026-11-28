"""14주차: 느린 층 LLM 보정 에이전트 (Claude API 도구 호출). 세션이 끝난 뒤 실행하고, 실시간 경로에는 없다.

하는 일
  1. 결정 로그 요약과 "잘못된 큐였어요" 피드백을 읽고, 이 사용자에게 잘못된 큐가 생기는 상황을 찾는다.
  2. 임계값을 바꾸면 어떻게 되는지 저장된 확률로 미리 재생해 본다(simulate_threshold).
  3. 작은 변경은 스스로 적용하고(set_threshold), 큰 변경은 대시보드에 승인을 요청한다(request_approval).
  4. 찾은 패턴을 config/user_memory.json에 메모해 두고, 다음 실행 때 먼저 읽는다.

보내는 것: 로그 요약과 피드백 기록만. 원시 가속도는 보내지 않는다.
자율 범위(코드로 강제, 프롬프트만 믿지 않음):
  - 0.6~0.9 안에서, 이번 실행 시작 값과 0.05 이내 → 스스로 적용 (변경 기록표에 "자율 적용")
  - 단, 재생해 본 동결 감지율이 현재보다 5%p 넘게 떨어지면 작은 변경이라도 스스로 적용하지 않음
    (0.8 → 0.85만으로 감지율이 80.6% → 51.5%로 떨어져서 넣은 안전장치)
  - 그보다 큰 변경이나 범위 밖 → 승인 요청만 가능. 0.5~0.95 밖은 승인 요청도 거부
API 키가 없거나 네트워크가 끊기면 아무것도 바꾸지 않는다(현재 임계값 유지). 엣지 동작에는 영향이 없다.

실행:     .venv\\Scripts\\python.exe src\\llm_agent.py
도구 확인: .venv\\Scripts\\python.exe src\\llm_agent.py --check   (LLM 없이 도구 결과만 출력)
"""
import argparse
import json
import statistics
import sys
import time
from collections import Counter

import anthropic

from agent_eval import evaluate, run_agent, window_rhythm
from calibrate import loso_scores
from data import ROOT
from records import (APPROVAL_PATH, CHANGES_PATH, CONFIG_PATH, DECISIONS_PATH, FEEDBACK_PATH, LOGS, append_jsonl,
                     apply_threshold, read_json, read_jsonl, write_json, write_jsonl)
from train import label_episodes

MODEL = "claude-opus-5-5"
MEMORY_PATH = ROOT / "config" / "user_memory.json"
RUNS_PATH = LOGS / "llm_runs.jsonl"          # LLM 실행 1회 = 1줄 (도구 호출, 최종 설명, 결과)
AUTO_RANGE = (0.6, 0.9)                      # 이 안에서만 스스로 적용
AUTO_STEP = 0.05                             # 이번 실행 시작 값에서 이만큼까지만 스스로 바꿈
MAX_DETECT_DROP = 5.0                        # %p: 재생 감지율이 이보다 더 떨어지면 자율 적용 금지
HARD_RANGE = (0.5, 0.95)                     # 승인 요청도 이 밖은 거부
DETECT_KEY = "동결 에피소드 감지율(%)"
MAX_TURNS = 15                               # 도구 호출 왕복 최대 횟수
MAX_NOTES = 20                               # 메모는 최근 20개만 유지

SYSTEM = f"""당신은 파킨슨병 보행 동결(FoG) 보조 장치 StepCue의 "느린 층" 보정 에이전트입니다.
장치의 빠른 층(엣지 규칙 상태 머신)은 3윈도우 평균 동결 확률이 임계값 이상이면 메트로놈 큐를 켭니다.
당신은 세션이 끝난 뒤 기록을 읽고, 이 사용자에게 맞게 임계값을 조정하고 패턴을 메모합니다.

판단 원칙
- 실제 동결을 놓치는 것(낙상 위험)이 잘못된 큐(불편함)보다 위험합니다. 오탐을 줄이려고 감지율을 크게 희생하지 마세요.
- 근거 없이 바꾸지 마세요. 피드백이 적거나(예: 2건 이하) 패턴이 불분명하면 바꾸지 않는 것이 맞는 결정입니다.
- 임계값을 바꾸기 전에 반드시 simulate_threshold로 현재 값과 새 값을 둘 다 재생해 비교하세요.
- 피드백 기록의 prob_avg(큐가 켜진 순간의 확률) 분포를 보세요. 잘못된 큐가 임계값 바로 위에 몰려 있으면 임계값을 조금
  올리는 것이 효과가 있고, 높은 확률에서도 생기면 임계값으로는 해결되지 않습니다(그 사실을 메모하세요).
- 자율 범위: {AUTO_RANGE[0]}~{AUTO_RANGE[1]} 안에서 이번 실행 시작 값과 {AUTO_STEP} 이내이고, 재생한 동결 감지율이
  현재보다 {MAX_DETECT_DROP:.0f}%p 넘게 떨어지지 않는 변경만 set_threshold로 직접 적용할 수 있습니다.
  그 밖의 변경은 request_approval로 사람에게 승인을 요청하세요(감지율 손실을 이유에 꼭 적으세요).
- 찾은 패턴, 시도한 변경과 그 결과는 update_memory로 짧게 남기세요. 다음 실행의 당신이 그 메모를 먼저 읽습니다.
- 재생 시뮬레이션은 Daphnet 녹화 데이터의 LOSO 확률로 계산한 것이라, 피드백 상황 자체를 재현하지는 못합니다.

모든 이유와 메모는 한국어로, 숫자를 넣어 짧게 쓰세요. 마지막 답변은 무엇을 보고 무엇을 결정했는지 3~5문장으로 요약하세요."""


def _fmt(wall: float | None) -> str | None:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall)) if wall else None


def _stats(xs: list) -> dict | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "min": round(min(xs), 3), "median": round(statistics.median(xs), 3), "max": round(max(xs), 3)}


# ---------- 세션 나누기 ----------
def sessions(decisions: list[dict]) -> list[dict]:
    """결정 로그를 세션(재생 1회)으로 나눈다: 이름(source)이 바뀌거나 데이터 시간이 뒤로 가면 새 세션."""
    out, cur, prev_t = [], None, None
    for d in decisions:
        t = d.get("t")
        if cur is None or d.get("source") != cur["source"] or (t is not None and prev_t is not None and t < prev_t - 1):
            cur, prev_t = {"source": d.get("source"), "recs": []}, None
            out.append(cur)
        cur["recs"].append(d)
        if t is not None:
            prev_t = t
    for s in out:
        s["wall_start"], s["wall_end"] = s["recs"][0]["wall"], s["recs"][-1]["wall"]
    return out


def feedback_in(sess: dict, feedback: list[dict]) -> list[dict]:
    """세션 중(끝나고 60초 안 포함)에 들어온 피드백."""
    return [f for f in feedback if sess["wall_start"] - 1 <= f["wall"] <= sess["wall_end"] + 60]


def current_subject() -> str | None:
    ss = sessions(read_jsonl(DECISIONS_PATH))
    return ss[-1]["source"][:3] if ss and ss[-1]["source"] else None


# ---------- 도구 ----------
def get_log_summary(last_n: int = 3) -> dict:
    """최근 세션들의 결정 로그 요약. 정답 레이블(truth)은 평가용이라 넣지 않는다."""
    feedback = read_jsonl(FEEDBACK_PATH)
    ss = sessions(read_jsonl(DECISIONS_PATH))
    rows = []
    for i, s in enumerate(ss[-max(1, last_n):], start=max(1, len(ss) - last_n + 1)):
        recs = s["recs"]
        ts = [r["t"] for r in recs if r.get("t") is not None]
        starts = [r for r in recs if r["action"] == "cue_start"]
        durs, open_t = [], None
        for r in recs:
            if r["action"] == "cue_start":
                open_t = r["t"]
            elif r["action"] == "cue_stop" and open_t is not None and r.get("t") is not None:
                durs.append(r["t"] - open_t)
                open_t = None
        rows.append({
            "session": i, "source": s["source"], "start": _fmt(s["wall_start"]),
            "logged_data_time_s": [round(min(ts), 1), round(max(ts), 1)] if ts else None,
            "thresholds_used": sorted({r["threshold"] for r in recs if r.get("threshold") is not None}),
            "action_counts": dict(Counter(r["action"] for r in recs)),
            "cue_start_prob": _stats([r.get("prob_avg") for r in starts]),
            "cue_start_walking_share": round(sum(bool(r.get("walking")) for r in starts) / len(starts), 2)
            if starts else None,
            "cue_start_rhythm": _stats([r.get("rhythm") for r in starts]),
            "cue_duration_s": _stats(durs),
            "no_action_prob": _stats([r.get("prob_avg") for r in recs if r["action"] == "no_action"]),
            "wrong_cue_feedback": len(feedback_in(s, feedback))})
    return {"total_sessions": len(ss), "sessions": rows,
            "note": "no_action = 확률이 (임계값 - 0.2) 이상이었지만 임계값 미만이라 행동하지 않은 판단"}


def get_feedback() -> dict:
    """"잘못된 큐였어요" 기록 전체와 그때의 상황. 큐가 켜진 순간의 값이다."""
    fb = read_jsonl(FEEDBACK_PATH)
    keys = ["source", "t_cue", "prob_avg", "threshold", "walking", "rhythm", "cadence", "bpm",
            "prev_action", "since_prev_s", "by", "note"]
    recs = [{"time": _fmt(f["wall"]), **{k: f.get(k) for k in keys}} for f in fb[-60:]]
    return {"count": len(fb), "shown": len(recs), "records": recs,
            "prob_avg": _stats([f.get("prob_avg") for f in fb]),
            "walking_share": round(sum(bool(f.get("walking")) for f in fb) / len(fb), 2) if fb else None,
            "by_threshold": dict(Counter(str(f.get("threshold")) for f in fb))}


_replay = {}


def simulate_threshold(threshold: float) -> dict:
    """저장된 LOSO 확률(Daphnet 10명)을 빠른 층 에이전트로 다시 재생해서 그 임계값의 결과를 계산한다."""
    threshold = round(float(threshold), 3)
    if threshold in _replay:
        return _replay[threshold]
    if "w" not in _replay:
        w = loso_scores().merge(window_rhythm(), on=["subject", "run", "t_end"], how="left", validate="1:1")
        _replay["w"] = w.sort_values(["subject", "run", "t_end"]).reset_index(drop=True)
        _replay["eps"] = label_episodes()
    w, eps = _replay["w"], _replay["eps"]
    cue_on, acts = run_agent(w, require_walking=False, threshold=threshold)
    m = evaluate(w, eps, cue_on, acts)
    res = {"threshold": threshold,
           "동결 에피소드 감지율(%)": round(m["에피소드 감지율"] * 100, 1),
           "감지 지연 중앙값(초)": round(m["감지 지연 중앙값(초)"], 2),
           "동결 없는 환자(S04·S10) 큐(10분당)": round(m["S04·S10 오탐(10분당)"], 2),
           "동결 없는 환자 보호자 알림(10분당)": round(m["S04·S10 보호자 알림(10분당)"], 2)}
    subj = current_subject()
    if subj and subj in set(w.subject):
        minutes = (w.subject == subj).sum() * 0.5 / 60
        n = ((acts.subject == subj) & (acts.action == "cue_start")).sum()
        res[f"이 사용자({subj}) 녹화 전체 큐(10분당)"] = round(n / minutes * 10, 2)
    _replay[threshold] = res
    return res


def _expected(threshold: float) -> dict | None:
    """기록표의 "예상 효과"는 LLM이 쓴 숫자가 아니라 실제 재생 결과를 넣는다."""
    r = _replay.get(round(float(threshold), 3))
    return {k: v for k, v in r.items() if k != "threshold"} if r else None


class Tools:
    """LLM이 부르는 도구. 자율 범위는 여기서 강제한다."""

    def __init__(self):
        self.start_thr = read_json(CONFIG_PATH)["threshold"]
        self.calls = []

    def set_threshold(self, threshold: float, reason: str) -> dict:
        thr = round(float(threshold), 3)
        if not (AUTO_RANGE[0] <= thr <= AUTO_RANGE[1]) or abs(thr - self.start_thr) > AUTO_STEP + 1e-9:
            raise ValueError(f"자율 범위 밖: {AUTO_RANGE[0]}~{AUTO_RANGE[1]} 안에서 시작 값 {self.start_thr}과 "
                             f"{AUTO_STEP} 이내만 직접 적용 가능. 더 큰 변경은 request_approval을 쓰세요.")
        if thr not in _replay:
            raise ValueError("먼저 simulate_threshold로 이 값을 재생해 보세요.")
        drop = simulate_threshold(self.start_thr)[DETECT_KEY] - _replay[thr][DETECT_KEY]
        if drop > MAX_DETECT_DROP:
            raise ValueError(f"재생 감지율이 {drop:.1f}%p 떨어집니다(허용 {MAX_DETECT_DROP:.0f}%p). "
                             "스스로 적용할 수 없으니 request_approval로 사람에게 물어보세요.")
        rec = apply_threshold(thr, reason, "자율 적용", _expected(thr))
        return {"applied": True, "from": rec["from"], "to": rec["to"]}

    def request_approval(self, threshold: float, reason: str) -> dict:
        thr = round(float(threshold), 3)
        if not (HARD_RANGE[0] <= thr <= HARD_RANGE[1]):
            raise ValueError(f"{HARD_RANGE[0]}~{HARD_RANGE[1]} 밖의 임계값은 안전을 위해 요청할 수 없습니다.")
        old = read_json(APPROVAL_PATH)
        if old and old.get("status") == "pending":
            raise ValueError(f"이미 승인을 기다리는 요청이 있습니다({old['from']} → {old['to']}).")
        if thr not in _replay:
            raise ValueError("먼저 simulate_threshold로 이 값을 재생해 보세요.")
        req = {"wall": round(time.time(), 3), "from": read_json(CONFIG_PATH)["threshold"], "to": thr,
               "reason": reason, "expected": _expected(thr), "status": "pending"}
        write_json(APPROVAL_PATH, req)
        return {"requested": True, "message": "대시보드에 승인 요청을 올렸습니다. 사람이 승인하면 적용됩니다."}

    def update_memory(self, note: str) -> dict:
        mem = read_json(MEMORY_PATH) or {"notes": []}
        mem["notes"] = (mem["notes"] + [{"time": _fmt(time.time()), "note": note}])[-MAX_NOTES:]
        write_json(MEMORY_PATH, mem)
        return {"saved": True, "notes": len(mem["notes"])}

    def run(self, name: str, args: dict) -> dict:
        fn = {"get_log_summary": get_log_summary, "get_feedback": get_feedback,
              "simulate_threshold": simulate_threshold, "set_threshold": self.set_threshold,
              "request_approval": self.request_approval, "update_memory": self.update_memory}[name]
        return fn(**args)


def _tool(name, desc, props):
    return {"name": name, "description": desc, "strict": True,
            "input_schema": {"type": "object", "properties": props, "required": list(props),
                             "additionalProperties": False}}


_THR = {"type": "number", "description": "새 임계값 (3윈도우 평균 동결 확률 기준, 예: 0.85)"}
_REASON = {"type": "string", "description": "바꾸는 이유 (피드백 패턴과 재생 결과 숫자 포함, 한국어)"}
TOOLS = [
    _tool("get_log_summary", "최근 세션들의 결정 로그 요약: 행동 횟수, 큐가 켜진 순간의 확률·걷기 비율, 큐 지속 시간, "
          "세션별 잘못된 큐 피드백 수.", {"last_n": {"type": "integer", "description": "최근 몇 세션 (보통 3)"}}),
    _tool("get_feedback", "\"잘못된 큐였어요\" 피드백 전체. 각 기록은 그 큐가 켜진 순간의 확률, 당시 임계값, 걷기 여부, "
          "걷기 리듬, 걸음 속도, 직전 행동을 담고 있다.", {}),
    _tool("simulate_threshold", "Daphnet 녹화 데이터(10명)의 저장된 확률을 이 임계값으로 다시 재생해서 동결 감지율, 감지 지연, "
          "동결 없는 환자의 큐 횟수, 이 사용자 녹화의 큐 횟수를 계산한다. 약 10초 걸린다.", {"threshold": _THR}),
    _tool("set_threshold", f"임계값을 바로 적용한다. {AUTO_RANGE[0]}~{AUTO_RANGE[1]} 안에서 이번 실행 시작 값과 "
          f"{AUTO_STEP} 이내만 가능하고, 먼저 simulate_threshold로 재생한 값이어야 한다.",
          {"threshold": _THR, "reason": _REASON}),
    _tool("request_approval", "자율 범위를 넘는 임계값 변경을 대시보드에 올려 사람의 승인을 요청한다. "
          "먼저 simulate_threshold로 재생한 값이어야 한다.", {"threshold": _THR, "reason": _REASON}),
    _tool("update_memory", "이 사용자에 대해 알아낸 패턴이나 결정을 메모한다. 다음 실행 때 먼저 읽힌다.",
          {"note": {"type": "string", "description": "한두 문장 메모 (한국어)"}}),
]


# ---------- 변경 후 실제 효과 ----------
def fill_after():
    """적용된 변경의 "적용 후 효과" 칸을 채운다: 변경 뒤 첫 세션의 큐 횟수와 잘못된 큐 피드백 수.
    LLM이 아니라 코드가 센다(숫자를 지어내지 않게)."""
    changes = read_jsonl(CHANGES_PATH)
    ss, fb = sessions(read_jsonl(DECISIONS_PATH)), read_jsonl(FEEDBACK_PATH)
    dirty = False
    for c in changes:
        if not c.get("applied") or c.get("after") is not None:
            continue
        nxt = next((s for s in ss if s["wall_start"] > c["wall"]), None)
        if nxt is None:
            continue
        cues = sum(r["action"] == "cue_start" for r in nxt["recs"])
        c["after"] = {"session": nxt["source"], "start": _fmt(nxt["wall_start"]),
                      "큐": cues, "잘못된 큐": len(feedback_in(nxt, fb))}
        dirty = True
    if dirty:
        write_jsonl(CHANGES_PATH, changes)


def opening_message() -> str:
    """첫 메시지: 현재 설정, 이전 메모, 변경 기록, 승인 요청 결과. 자세한 기록은 도구로 읽게 한다."""
    cfg = read_json(CONFIG_PATH)
    mem = read_json(MEMORY_PATH) or {"notes": []}
    changes = [{"time": _fmt(c["wall"]), **{k: c.get(k) for k in ["from", "to", "approval", "applied", "after",
                                                                   "reason"]}}
               for c in read_jsonl(CHANGES_PATH)[-5:]]
    appr = read_json(APPROVAL_PATH)
    parts = [f"현재 설정: {json.dumps(cfg, ensure_ascii=False)}",
             "이전 실행의 메모:\n" + ("\n".join(f"- [{n['time']}] {n['note']}" for n in mem["notes"]) or "(없음)"),
             "최근 임계값 변경 기록(after = 변경 뒤 첫 세션의 실제 결과):\n"
             + ("\n".join(json.dumps(c, ensure_ascii=False) for c in changes) or "(없음)")]
    if appr:
        parts.append(f"마지막 승인 요청: {appr['from']} → {appr['to']}, 상태 = {appr['status']}")
    parts.append("기록을 읽고 이 사용자의 임계값을 바꿀지 결정하세요.")
    return "\n\n".join(parts)


# ---------- 실행 ----------
def run(max_turns: int = MAX_TURNS) -> dict:
    fill_after()
    tools = Tools()
    log = {"wall": round(time.time(), 3), "model": MODEL, "start_threshold": tools.start_thr, "tool_calls": [],
           "status": None, "final_text": None, "usage": {"input": 0, "output": 0}}
    messages = [{"role": "user", "content": opening_message()}]
    try:
        client = anthropic.Anthropic()
        for _ in range(max_turns):
            resp = client.beta.messages.create(
                model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=messages,
                output_config={"effort": "high"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default")   # 거절 시 다른 모델로 자동 재시도
            log["usage"]["input"] += resp.usage.input_tokens
            log["usage"]["output"] += resp.usage.output_tokens
            if resp.stop_reason == "refusal":
                log["status"] = "refusal"
                break
            messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                log["status"] = "ok"
                log["final_text"] = "\n".join(b.text for b in resp.content if b.type == "text")
                break
            results = []
            for b in uses:
                print(f"  도구: {b.name}({json.dumps(b.input, ensure_ascii=False)})", flush=True)
                try:
                    out, err = tools.run(b.name, b.input), False
                except (ValueError, TypeError, KeyError) as e:
                    out, err = {"error": str(e)}, True
                print(f"    → {json.dumps(out, ensure_ascii=False)[:300]}", flush=True)
                log["tool_calls"].append({"name": b.name, "input": b.input, "error": err,
                                          "output": out if b.name not in ("get_log_summary", "get_feedback") else
                                          "(생략)"})
                results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": err,
                                "content": json.dumps(out, ensure_ascii=False)})
            messages.append({"role": "user", "content": results})
        else:
            log["status"] = "max_turns"
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.APIConnectionError,
            anthropic.RateLimitError, anthropic.APIStatusError, TypeError) as e:
        # TypeError: 인증 정보가 전혀 없을 때 SDK가 내는 오류
        log["status"] = "offline"
        log["final_text"] = f"LLM을 쓸 수 없어 현재 임계값을 유지합니다 ({type(e).__name__}: {str(e)[:150]})"
    log["end_threshold"] = read_json(CONFIG_PATH)["threshold"]
    append_jsonl(RUNS_PATH, log)
    return log


def check():
    """LLM 없이 도구 결과만 출력 (API 키 없이 도구가 제대로 동작하는지 확인)."""
    fill_after()
    print("== get_log_summary ==\n" + json.dumps(get_log_summary(3), ensure_ascii=False, indent=1))
    print("== get_feedback ==\n" + json.dumps(get_feedback(), ensure_ascii=False, indent=1)[:2000])
    thr = read_json(CONFIG_PATH)["threshold"]
    for t in (thr, round(thr + AUTO_STEP, 3)):
        t0 = time.time()
        res = simulate_threshold(t)
        print(f"== simulate_threshold({t}) ({time.time() - t0:.1f}초) ==")
        print(json.dumps(res, ensure_ascii=False, indent=1))
    print("== 첫 메시지 ==\n" + opening_message())


def main():
    sys.stdout.reconfigure(errors="replace")      # 윈도우 콘솔(cp949)에 없는 기호가 있어도 멈추지 않게
    ap = argparse.ArgumentParser(description="느린 층 LLM 보정 에이전트 (세션이 끝난 뒤 실행)")
    ap.add_argument("--check", action="store_true", help="LLM 없이 도구 결과만 출력")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    a = ap.parse_args()
    if a.check:
        check()
        return
    print(f"LLM 에이전트 시작 (모델 {MODEL})", flush=True)
    log = run(a.max_turns)
    print(f"\n상태: {log['status']} · 임계값 {log['start_threshold']} → {log['end_threshold']} · "
          f"토큰 입력 {log['usage']['input']} / 출력 {log['usage']['output']}")
    if log["final_text"]:
        print(log["final_text"])


if __name__ == "__main__":
    main()
