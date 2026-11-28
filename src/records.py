"""14주차: 기록 파일 경로와 읽기·쓰기 (대시보드, 빠른 층 에이전트, LLM 에이전트가 같이 사용).

logs/decisions.jsonl          빠른 층 에이전트의 판단 (src/agent.py가 씀)
logs/feedback.jsonl           "잘못된 큐였어요" 피드백 (대시보드 버튼 또는 가상 사용자)
logs/threshold_changes.jsonl  임계값 변경 기록표 (LLM 에이전트가 씀, 대시보드가 표시)
logs/pending_approval.json    LLM 에이전트의 승인 요청 1건 (대시보드에서 승인/거절하면 status가 바뀜)
"""
import json
import time
from pathlib import Path

from data import ROOT

LOGS = ROOT / "logs"
DECISIONS_PATH = LOGS / "decisions.jsonl"
FEEDBACK_PATH = LOGS / "feedback.jsonl"
CHANGES_PATH = LOGS / "threshold_changes.jsonl"
APPROVAL_PATH = LOGS / "pending_approval.json"
CONFIG_PATH = ROOT / "config" / "user.json"


def append_jsonl(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, recs: list[dict]):
    """jsonl 전체를 다시 쓴다 (변경 기록표의 "적용 후 효과" 칸을 나중에 채울 때)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")                 # 반쯤 쓴 파일을 다른 프로그램이 읽지 않게 바꿔치기
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def record_feedback(cue: dict, by: str, source: str | None = None, prev_action: dict | None = None,
                    note: str | None = None) -> dict:
    """"잘못된 큐였어요" 피드백 1건. cue = 그 큐의 cue_start 행동(엣지 에이전트가 보낸 메시지).

    LLM 에이전트가 "어떤 상황에서 잘못된 큐가 생기는지" 찾을 수 있게, 큐가 켜진 순간의 상황을 같이 남긴다.
    원시 가속도는 넣지 않는다.
    """
    rec = {"wall": round(time.time(), 3), "by": by, "source": source,
           "t_cue": cue.get("t"), "prob_avg": cue.get("prob_avg"), "threshold": cue.get("threshold"),
           "walking": cue.get("walking"), "rhythm": cue.get("rhythm"), "cadence": cue.get("cadence"),
           "bpm": cue.get("bpm"),
           "prev_action": prev_action.get("action") if prev_action else None,
           "since_prev_s": round(cue["t"] - prev_action["t"], 1)
           if prev_action and prev_action.get("t") is not None and cue.get("t") is not None else None,
           "note": note}
    append_jsonl(FEEDBACK_PATH, rec)
    return rec


def record_change(old: float, new: float, reason: str, approval: str, applied: bool,
                  expected: dict | None = None, by: str = "llm_agent") -> dict:
    """임계값 변경 기록표에 1줄 추가. approval: "자율 적용" / "사람 승인" / "거절"."""
    rec = {"wall": round(time.time(), 3), "by": by, "from": old, "to": new, "reason": reason,
           "expected": expected, "approval": approval, "applied": applied,
           "after": None}            # 다음 세션의 실제 효과(잘못된 큐 횟수)는 LLM 에이전트가 나중에 채움
    append_jsonl(CHANGES_PATH, rec)
    return rec


def apply_threshold(new: float, reason: str, approval: str, expected: dict | None = None,
                    by: str = "llm_agent") -> dict:
    """config/user.json의 임계값을 바꾸고 기록한다. 엣지 노드가 파일 변경을 감지해 바로 다시 읽는다."""
    cfg = read_json(CONFIG_PATH)
    old = cfg["threshold"]
    cfg["threshold"] = round(float(new), 3)
    write_json(CONFIG_PATH, cfg)
    return record_change(old, cfg["threshold"], reason, approval, True, expected, by)
