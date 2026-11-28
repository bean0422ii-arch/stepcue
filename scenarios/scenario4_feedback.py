"""14주차: 테스트 시나리오 ④ 오탐 피드백 학습 (빠른 층 + 느린 층 LLM 에이전트 전체 순환).

순서
  1회차  동결이 없는 S04 R01(51분)을 실제 엣지 코드로 재생. 가상 사용자가 대시보드와 같은 규칙으로
         "잘못된 큐였어요"를 누른다(큐가 켜졌다 꺼질 때까지 정답 레이블에 동결이 없으면).
  LLM    src/llm_agent.py가 기록을 읽고 임계값을 바꾸거나(자율 범위 안) 승인을 요청한다(Claude API).
  승인   승인 요청이 있으면 사람이 터미널에서 y/n (--approve면 승인한 것으로 진행)
  2회차  같은 S04 R01을 바뀐 임계값으로 다시 재생 → 잘못된 큐가 줄었는지
  안전   바뀐 임계값으로 시나리오 ①②(동결 환자) 구간을 다시 재생 + LOSO 재생 감지율 → 동결을 놓치지 않는지

실제 config/와 logs/는 건드리지 않는다. 기록은 모두 scenarios/4_feedback/ 안에 쌓인다.
API 키가 없으면 LLM 단계가 offline으로 끝나고 임계값이 그대로라 "감소" 기준은 실패로 나온다.

실행: .venv\\Scripts\\python.exe scenarios\\scenario4_feedback.py [--approve]   (Claude API 사용, 약 $0.3)
결과: scenarios/4_feedback/results.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import edge_node                                # noqa: E402
import llm_agent as L                           # noqa: E402
import records as R                             # noqa: E402
import run_scenarios as RS                      # noqa: E402
from agent import Agent                         # noqa: E402
from simulator import replay                    # noqa: E402

OUT = HERE / "4_feedback"
SUBJECT, RUN = "S04", "R01"
START_THR = 0.8


def use_scenario_paths():
    """기록 경로를 scenarios/4_feedback/ 안으로 바꾼다 (모듈마다 import한 이름도 같이)."""
    paths = {"DECISIONS_PATH": OUT / "logs" / "decisions.jsonl", "FEEDBACK_PATH": OUT / "logs" / "feedback.jsonl",
             "CHANGES_PATH": OUT / "logs" / "threshold_changes.jsonl",
             "APPROVAL_PATH": OUT / "logs" / "pending_approval.json", "CONFIG_PATH": OUT / "config" / "user.json"}
    for name, p in paths.items():
        setattr(R, name, p)
        if hasattr(L, name):
            setattr(L, name, p)
    edge_node.CONFIG_PATH = paths["CONFIG_PATH"]
    L.MEMORY_PATH = OUT / "config" / "user_memory.json"
    L.RUNS_PATH = OUT / "logs" / "llm_runs.jsonl"
    RS.LOG_DIR = OUT / "logs"
    for d in ("logs", "config"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
        for f in (OUT / d).iterdir():                      # 지난 실행 기록 지우기 (이 폴더 안의 파일만)
            f.unlink()
    R.write_json(paths["CONFIG_PATH"], {"threshold": START_THR, "smooth_windows": 3})


def replay_with_virtual_user(label: str) -> dict:
    """S04 R01 전체 재생. 가상 사용자 규칙은 dashboard/app.py와 같다."""
    node, agent = edge_node.EdgeNode(), Agent(log_path=R.DECISIONS_PATH)
    agent.reset(source=f"{SUBJECT}{RUN}")
    cue, saw_fog, last_real, prev_action = None, False, None, None
    cues, wrong = 0, 0
    for s in replay(SUBJECT, RUN, 0, None, speed=1e9):
        ev = node.on_sample(s)
        if ev is None:
            continue
        if cue is not None and ev.get("truth") == 2:
            saw_fog = True
        for a in agent.step(ev):
            if a["action"] == "cue_start":
                cue, saw_fog, prev_action = a, a.get("truth") == 2, last_real
                cues += 1
            elif a["action"] == "cue_stop" and cue is not None:
                if not saw_fog:
                    R.record_feedback(cue, by="가상 사용자", source=cue.get("source"), prev_action=prev_action,
                                      note="큐가 켜진 동안 정답 레이블에 동결 없음")
                    wrong += 1
                cue = None
            if a["action"] != "no_action":
                last_real = a
    minutes = s["t"] / 60                         # 마지막 샘플 시각
    print(f"[{label}] 임계값 {node.cfg['threshold']}: 큐 {cues}회, 잘못된 큐 {wrong}회 ({minutes:.0f}분)", flush=True)
    return {"threshold": node.cfg["threshold"], "cues": cues, "wrong": wrong, "minutes": round(minutes, 1)}


def handle_approval(auto: bool) -> str | None:
    req = R.read_json(R.APPROVAL_PATH)
    if not req or req.get("status") != "pending":
        return None
    print(f"\n승인 요청: 임계값 {req['from']} → {req['to']}\n이유: {req['reason']}\n예상 효과: {req['expected']}")
    yes = auto or input("승인할까요? (y/n) ").strip().lower() == "y"
    if yes:
        R.apply_threshold(req["to"], req["reason"], "사람 승인", req.get("expected"))
    else:
        R.record_change(req["from"], req["to"], req["reason"], "거절", False, req.get("expected"))
    R.write_json(R.APPROVAL_PATH, {**req, "status": "approved" if yes else "rejected", "decided": round(time.time(), 3),
                                   "by": "--approve 옵션" if auto else "터미널 입력"})
    print("→ 승인" if yes else "→ 거절")
    return "승인" if yes else "거절"


def safety_check() -> list[dict]:
    """바뀐 임계값으로 시나리오 ①②(동결 환자) 구간을 다시 재생."""
    rows = []
    for sc in RS.SCENARIOS[:2]:
        res = RS.run_segment(sc)
        checks, info = RS.check(sc, res)
        rows.append({"title": sc["title"], "ok": all(p for _, p, _ in checks),
                     "actions": info["기록된 행동"], "checks": checks})
    return rows


def main():
    ap = argparse.ArgumentParser(description="시나리오 ④ 오탐 피드백 학습")
    ap.add_argument("--approve", action="store_true", help="승인 요청이 오면 자동으로 승인 (기본: 터미널에서 묻기)")
    a = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    use_scenario_paths()

    run1 = replay_with_virtual_user("1회차")
    print("\nLLM 에이전트 실행 중...", flush=True)
    log = L.run()
    print(f"LLM 상태: {log['status']} · 임계값 {log['start_threshold']} → {log['end_threshold']}\n{log['final_text']}")
    approval = handle_approval(a.approve)
    new_thr = R.read_json(R.CONFIG_PATH)["threshold"]

    time.sleep(1.1)                                 # 새 세션이 확실히 뒤에 오도록
    run2 = replay_with_virtual_user("2회차")
    L.fill_after()                                  # 변경 기록표의 "적용 후 효과" 채우기
    safety = safety_check()
    sim_old, sim_new = L.simulate_threshold(START_THR), L.simulate_threshold(new_thr)
    drop = sim_old[L.DETECT_KEY] - sim_new[L.DETECT_KEY]

    checks = [("LLM이 실제로 실행됨 (Claude API)", log["status"] == "ok", log["status"]),
              ("임계값 변경이 기록표에 남음", len(R.read_jsonl(R.CHANGES_PATH)) > 0,
               f"{START_THR} → {new_thr}"),
              ("2회차 잘못된 큐 감소", run2["wrong"] < run1["wrong"], f"{run1['wrong']} → {run2['wrong']}회"),
              ("동결 환자 구간 ①② 여전히 통과", all(r["ok"] for r in safety),
               ", ".join(f"{r['title'][:1]} {'통과' if r['ok'] else '실패'}" for r in safety)),
              ("LOSO 재생 감지율 하락 ≤ 5%p", drop <= L.MAX_DETECT_DROP,
               f"{sim_old[L.DETECT_KEY]}% → {sim_new[L.DETECT_KEY]}% ({-drop:+.1f}%p)")]
    for n, p, v in checks:
        print(f"   {'O' if p else 'X'} {n}: {v}")
    write_report(run1, run2, log, approval, safety, sim_old, sim_new, checks)


def write_report(run1, run2, log, approval, safety, sim_old, sim_new, checks):
    ok = all(p for _, p, _ in checks)
    changes = R.read_jsonl(R.CHANGES_PATH)
    mem = R.read_json(L.MEMORY_PATH) or {"notes": []}
    fb = R.read_jsonl(R.FEEDBACK_PATH)
    md = ["# 시나리오 ④ 오탐 피드백 학습 결과", "",
          f"실행: `.venv\\Scripts\\python.exe scenarios\\scenario4_feedback.py` · {time.strftime('%Y-%m-%d %H:%M')} · "
          f"모델 `{log['model']}` · 결과 **{'통과' if ok else '실패'}**", "",
          "동결이 없는 S04 R01(51분)을 실제 엣지 코드로 두 번 재생했다. 1회차에서 가상 사용자가 \"잘못된 큐였어요\"를 "
          "누르고(대시보드와 같은 규칙: 큐가 켜졌다 꺼질 때까지 정답 레이블에 동결이 없으면), LLM 에이전트가 그 기록만 읽고 "
          "임계값을 정했다. 원시 가속도는 보내지 않았다.", "",
          "| 통과 기준 | 결과 | 측정 |", "|---|---|---|"]
    md += [f"| {n} | {'통과' if p else '실패'} | {v} |" for n, p, v in checks]
    md += ["", "## 1회차 → 2회차", "", "| | 임계값 | 큐 | 잘못된 큐 | 잘못된 큐(10분당) |", "|---|---|---|---|---|"]
    for name, r in (("1회차", run1), ("2회차", run2)):
        md.append(f"| {name} | {r['threshold']} | {r['cues']} | {r['wrong']} | {r['wrong'] / r['minutes'] * 10:.2f} |")
    probs = sorted(round(f["prob_avg"], 3) for f in fb if f["threshold"] == START_THR)
    md += ["", f"1회차 잘못된 큐가 켜진 순간의 확률: {probs}", "",
           "## LLM 에이전트가 한 일", "",
           f"상태 `{log['status']}`, 도구 호출 {len(log['tool_calls'])}번, 토큰 입력 {log['usage']['input']:,} / "
           f"출력 {log['usage']['output']:,}" + (f", 승인 요청 → {approval}" if approval else ""), "",
           "| # | 도구 | 입력 | 결과 |", "|---|---|---|---|"]
    for i, c in enumerate(log["tool_calls"], 1):
        out = c["output"] if isinstance(c["output"], str) else json.dumps(c["output"], ensure_ascii=False)
        md.append(f"| {i} | `{c['name']}` | {json.dumps(c['input'], ensure_ascii=False)} | "
                  f"{'오류: ' if c['error'] else ''}{out[:220]} |")
    md += ["", "**LLM의 마지막 설명**", "", "> " + (log["final_text"] or "(없음)").replace("\n", "\n> "), "",
           "**임계값 변경 기록표** (`logs/threshold_changes.jsonl`)", "",
           "| 변경 | 승인 | 적용 | 예상 효과(재생) | 적용 후 효과(2회차) | 이유 |", "|---|---|---|---|---|---|"]
    for c in changes:
        md.append(f"| {c['from']} → {c['to']} | {c['approval']} | {c['applied']} | "
                  f"{json.dumps(c['expected'], ensure_ascii=False)} | {json.dumps(c['after'], ensure_ascii=False)} | "
                  f"{c['reason']} |")
    md += ["", "**사용자 메모** (`config/user_memory.json`, 다음 실행 때 LLM이 먼저 읽음)", ""]
    md += [f"- {n['note']}" for n in mem["notes"]] or ["- (없음)"]
    md += ["", "## 안전 확인: 동결을 놓치지 않는가", "",
           f"- LOSO 재생(10명, 새 사람 기준): 감지율 {sim_old[L.DETECT_KEY]}% → {sim_new[L.DETECT_KEY]}%, "
           f"감지 지연 {sim_old['감지 지연 중앙값(초)']} → {sim_new['감지 지연 중앙값(초)']}초"]
    md += [f"- {r['title']} (바뀐 임계값으로 재생): {'통과' if r['ok'] else '실패'} ({r['actions']})" for r in safety]
    md += ["", "## 한계", "",
           "- S04 녹화로 임계값을 정하고 같은 S04 녹화로 효과를 확인했다. 실제보다 좋게 나올 수 있다.",
           "- 엣지 모델은 S04를 포함한 10명으로 학습했다. 새 사람 기준 성능은 LOSO 재생 숫자를 본다.",
           "- 피드백은 정답 레이블로 만든 가상 사용자다. 실제 사용자는 놓치거나 잘못 누를 수 있다.",
           "- 임계값은 이 사용자 전용 설정이다. 동결 환자 감지율 확인은 \"이 사용자에게 동결이 생기면\"을 다른 환자 "
           "데이터로 대신 본 것이다.", "",
           "전체 기록: [logs/](logs/), [config/](config/)", ""]
    (OUT / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"→ {OUT / 'results.md'}")


if __name__ == "__main__":
    main()
