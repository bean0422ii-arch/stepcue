"""14주차: 테스트 시나리오 ①~③. 실제 엣지 코드(특징 추출 → ONNX 추론 → 상태 머신)로 녹화 구간을 재생하고,
결정 로그와 통과/실패를 남긴다. 시나리오 ④(LLM 피드백 학습)는 API 키가 필요해서 따로 만든다.

MQTT 없이 한 프로세스 안에서 최대 속도로 재생한다(같은 EdgeNode, Agent 코드). 그래서 몇 초 안에 끝나고 결과가
항상 같다. 영상용 실시간 데모는 결과 문서의 "실시간 데모" 명령(브로커 + 엣지 노드 + 대시보드 + 시뮬레이터)으로 같은
구간을 재생하면 된다. MQTT를 거치면 센서 → 판단이 2~4ms 더 걸린다(docs/edge_metrics.md).

시간 두 가지
  감지 → 큐: 윈도우를 채우는 마지막 샘플을 받은 순간부터 큐 행동이 나올 때까지의 처리 시간(특징 + 추론 + 판단).
             통과 기준(problem card: < 1초)
  동결 시작 → 큐: 정답 레이블의 동결 시작부터 큐까지(데이터 시간). 2초 윈도우가 동결로 차는 시간이 대부분이라
             기준이 아니라 참고로 보고한다.

한계: 엣지 모델(models/rf.onnx)은 10명 전체로 학습했으므로, 이 재생은 학습에 쓴 데이터다. 새 사람에 대한 성능은
LOSO 결과(docs/model_results.md, docs/agent_results.md)를 본다. 시나리오는 "에이전트가 상황마다 올바른 행동을
하는지"를 확인하는 것이다.

실행: .venv\\Scripts\\python.exe scenarios\\run_scenarios.py   → scenarios/results.md, scenarios/logs/*.jsonl
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from agent import Agent                      # noqa: E402
from edge_node import CONFIG_PATH, EdgeNode  # noqa: E402
from simulator import replay                 # noqa: E402

LOG_DIR = HERE / "logs"
REPORT = HERE / "results.md"
LIVE = (".venv\\Scripts\\python.exe src\\simulator.py --subject {subject} --run {run} --start {start} "
        "--duration {duration} --mqtt")

SCENARIOS = [
    {"id": "1_short_fog", "title": "① 동결 → 큐 → 회복 → 중지",
     "subject": "S05", "run": "R01", "start": 686, "duration": 27,   # 앞 15초 걷기로 개인 걸음 속도를 배움
     "story": "2.5초짜리 짧은 동결. 앞에서 걷던 걸음 속도로 동결 중에 큐가 켜지고, 걷기가 회복되면 강화 없이 꺼져야 한다.",
     "expect": "`cue_start` → `cue_stop` (강화·알림 없음)"},
    {"id": "2_long_fog", "title": "② 긴 동결 → 큐 강화 → 보호자 알림",
     "subject": "S02", "run": "R01", "start": 936, "duration": 28,   # 앞의 동결(923.6~934.6초)이 끝난 뒤부터
     "story": "14초짜리 긴 동결. 3초 안에 회복이 없으면 템포를 90%로 낮추고 진동을 더하고, 10초가 넘으면 보호자에게 알려야 한다.",
     "expect": "`cue_start` → `cue_escalate`(템포 90%, 진동) → `notify_caregiver` (동결이 끝나기 전에)"},
    {"id": "3_no_action", "title": "③ 동결 없는 사람이 걷다 서기 반복 → 행동하지 않음",
     "subject": "S04", "run": "R01", "start": 1345, "duration": 60,
     "story": "동결이 없는 S04가 60초 동안 걷다 서기를 반복한다. 확률이 임계값 근처까지 오르지만 넘지 않으므로 "
              "큐를 켜지 않고, \"왜 행동하지 않았는지\"를 기록해야 한다.",
     "expect": "`no_action`만 (이유 기록), `cue_start` 0회"},
]


def run_segment(sc: dict) -> dict:
    """구간 하나를 재생한다. 엣지 노드와 같은 순서: on_sample → agent.step."""
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{sc['id']}.jsonl"
    log_path.unlink(missing_ok=True)
    node, agent = EdgeNode(), Agent(log_path=log_path)
    agent.reset(source=f"{sc['subject']}{sc['run']}")
    labels, events, actions = [], [], []
    for s in replay(sc["subject"], sc["run"], sc["start"], sc["duration"], speed=1e9):   # 기다리지 않고 재생
        labels.append((s["t"], s["label"]))
        t1 = time.perf_counter()
        ev = node.on_sample(s)
        if ev is None:
            continue
        acts = agent.step(ev)
        edge_ms = (time.perf_counter() - t1) * 1000               # 샘플 도착 → 판단까지
        events.append(ev)
        for a in acts:
            actions.append({**a, "edge_ms": round(edge_ms, 2)})
    return {"labels": labels, "events": events, "actions": actions, "log": log_path}


def fog_episodes(labels: list) -> list[tuple[float, float]]:
    """정답 레이블 2(동결)가 이어지는 구간들 (시작, 끝)."""
    eps, start, prev_t = [], None, None
    for t, y in labels:
        if y == 2 and start is None:
            start = t
        if y != 2 and start is not None:
            eps.append((start, prev_t))
            start = None
        prev_t = t
    if start is not None:
        eps.append((start, prev_t))
    return eps


def check(sc: dict, res: dict) -> tuple[list[tuple[str, bool, str]], dict]:
    """통과 기준 확인. (항목, 통과 여부, 측정값) 목록과 요약 숫자."""
    acts, eps = res["actions"], fog_episodes(res["labels"])
    names = [a["action"] for a in acts if a["action"] != "no_action"]
    first = {n: next((a for a in acts if a["action"] == n), None)
             for n in ["cue_start", "cue_escalate", "notify_caregiver", "cue_stop"]}
    near = [a for a in acts if a["action"] == "no_action"]
    max_p = max(e["prob_avg"] for e in res["events"])
    thr = res["events"][0]["threshold"]
    info = {"동결 구간(레이블)": ", ".join(f"{a:.1f}~{b:.1f}초 ({b - a:.1f}초)" for a, b in eps) or "없음",
            "최대 확률(3윈도우 평균)": f"{max_p:.3f} (임계값 {thr})",
            "기록된 행동": " → ".join(names) or "없음", "`no_action` 기록": f"{len(near)}회"}
    walk = [e["walking"] for e in res["events"]]
    info["걷기 ↔ 멈춤 전환"] = (f"{sum(a != b for a, b in zip(walk, walk[1:]))}회 "
                           f"(걷는 중 윈도우 {sum(walk) / len(walk):.0%})")
    c = []
    cs = first["cue_start"]
    if cs:
        info["감지 → 큐 (처리 시간)"] = f"{cs['edge_ms']:.2f} ms"
    if sc["id"] in ("1_short_fog", "2_long_fog"):
        ep = eps[0]
        c.append(("큐가 동결 중에 켜짐", cs is not None and ep[0] <= cs["t"] <= ep[1] + 0.5,
                  f"t={cs['t']:.1f}초" if cs else "큐 없음"))
        if cs:
            info["동결 시작 → 큐 (참고)"] = f"{cs['t'] - ep[0]:.2f} 초"
            c.append(("감지 → 큐 < 1초", cs["edge_ms"] < 1000, f"{cs['edge_ms']:.2f} ms"))
    if sc["id"] == "1_short_fog":
        st = first["cue_stop"]
        c.append(("회복 뒤 큐 중지", st is not None and st["t"] >= eps[0][1] - 0.5,
                  f"동결 끝 {st['t'] - eps[0][1]:+.1f}초" if st else "중지 없음"))
        c.append(("강화·알림 없음", first["cue_escalate"] is None and first["notify_caregiver"] is None,
                  "없음" if first["cue_escalate"] is None else "있음"))
    if sc["id"] == "2_long_fog":
        es, nt = first["cue_escalate"], first["notify_caregiver"]
        c.append(("3초 뒤 강화 (템포 90%, 진동)", es is not None and es.get("vibrate") is True
                  and es["bpm"] == round(cs["bpm"] * 0.9),
                  f"큐 후 {es['t'] - cs['t']:.1f}초, {cs['bpm']} → {es['bpm']}bpm" if es else "강화 없음"))
        c.append(("동결 중 보호자 알림", nt is not None and nt["t"] <= eps[0][1],
                  f"큐 후 {nt['t'] - cs['t']:.1f}초 (동결 끝 {nt['t'] - eps[0][1]:+.1f}초)" if nt else "알림 없음"))
        c.append(("순서: 시작 → 강화 → 알림", names[:3] == ["cue_start", "cue_escalate", "notify_caregiver"],
                  " → ".join(names[:3])))
    if sc["id"] == "3_no_action":
        c.append(("`cue_start` 0회", first["cue_start"] is None, f"{names.count('cue_start')}회"))
        c.append(("확률이 올랐을 때 `no_action`과 이유 기록", len(near) > 0, f"{len(near)}회"))
    return c, info


def timeline(res: dict, sc: dict) -> str:
    """결정 로그를 표로. ①②는 no_action을 빼고(개수는 위에 적음), ③은 처음 몇 개와 확률이 가장 높았던 것만."""
    acts = [a for a in res["actions"] if a["action"] != "no_action"]
    if sc["id"] == "3_no_action":
        acts = res["actions"]
        near = [a for a in acts if a["action"] == "no_action"]
        acts = near[:4] + sorted(near[4:], key=lambda a: -a["prob_avg"])[:2]
        acts.sort(key=lambda a: a["t"])
    rows = ["| t(초) | 행동 | 확률 | 걷기 | 정답 | 이유 |", "|---|---|---|---|---|---|"]
    truth = {0: "-", 1: "정상", 2: "동결", None: "-"}
    for a in acts:
        rows.append(f"| {a['t']:.1f} | `{a['action']}`{' ' + str(a['bpm']) + 'bpm' if a.get('bpm') else ''}"
                    f"{' +진동' if a.get('vibrate') else ''} | {a['prob_avg']:.3f} | {'O' if a['walking'] else 'X'} | "
                    f"{truth[a.get('truth')]} | {a['reason']} |")
    return "\n".join(rows)


def main():
    sys.stdout.reconfigure(errors="replace")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    out = ["# 테스트 시나리오 결과 (①~③)", "",
           f"만든 명령: `.venv\\Scripts\\python.exe scenarios\\run_scenarios.py` · 실행 {time.strftime('%Y-%m-%d %H:%M')} · "
           f"임계값 {cfg['threshold']} · 결정 로그 전체는 [logs/](logs/)", "",
           "실제 엣지 코드(특징 추출 → ONNX 추론 → 상태 머신)로 녹화 구간을 재생했다. MQTT 없이 한 프로세스에서 "
           "최대 속도로 돌려서 결과가 항상 같다. 엣지 모델은 10명 전체로 학습했으므로 이 구간들은 학습에 쓴 데이터다. "
           "새 사람에 대한 성능은 LOSO 결과([agent_results.md](../docs/agent_results.md))를 본다.", ""]
    summary = ["| 시나리오 | 구간 | 결과 |", "|---|---|---|"]
    body = []
    for sc in SCENARIOS:
        res = run_segment(sc)
        checks, info = check(sc, res)
        ok = all(p for _, p, _ in checks)
        print(f"{sc['title']}: {'통과' if ok else '실패'}", flush=True)
        for name, p, val in checks:
            print(f"   {'O' if p else 'X'} {name}: {val}", flush=True)
        seg = f"{sc['subject']} {sc['run']} {sc['start']}~{sc['start'] + sc['duration']}초"
        summary.append(f"| {sc['title']} | {seg} | {'**통과**' if ok else '**실패**'} |")
        body += [f"## {sc['title']}", "", sc["story"], "",
                 f"- 구간: {seg}", f"- 기대 행동: {sc['expect']}"]
        body += [f"- {k}: {v}" for k, v in info.items()]
        body += ["", "| 통과 기준 | 결과 | 측정 |", "|---|---|---|"]
        body += [f"| {n} | {'통과' if p else '실패'} | {v} |" for n, p, v in checks]
        body += ["", "**결정 로그**" + (" (no_action 중 처음 4개 + 확률이 가장 높았던 2개)" if sc["id"] == "3_no_action"
                                       else " (no_action 제외)"), "", timeline(res, sc), "",
                 f"전체 기록: [logs/{res['log'].name}](logs/{res['log'].name})", "",
                 "실시간 데모 (브로커, 엣지 노드, 대시보드를 먼저 실행): `"
                 + LIVE.format(**{k: sc[k] for k in ("subject", "run", "start", "duration")}) + "`", ""]
    REPORT.write_text("\n".join(out + summary + [""] + body), encoding="utf-8")
    print(f"→ {REPORT}")


if __name__ == "__main__":
    main()
