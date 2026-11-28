"""14주차: 에이전트 오프라인 평가. "직전 5초 걷는 중" 조건이 오탐을 얼마나 줄이는가?

방법
1. 확률: 13주차와 같은 LOSO 확률 + 3윈도우 평균 (data/processed/loso_scores.csv.gz, src/calibrate.py가 만듦)
2. 걷기 여부·걸음 속도: 엣지 노드와 똑같이, 윈도우 끝까지의 최근 10초 발목 가속도로 계산(처음 한 번 계산 후 저장)
3. 윈도우를 시간 순서대로 src/agent.py의 Agent에 넣는다(엣지 이벤트와 같은 형식). 임계값 0.8 고정
4. 비교
   - v1 알람: 확률 ≥ 0.8이면 바로 알람 (12주차 방식)
   - 에이전트(조건 없음): 상태 머신만 (회복 판단 2윈도우)
   - 에이전트(직전 5초 걷는 중): 상태 머신 + 걷기 조건 (가설: 일부러 멈춘 것을 거르면 오탐이 준다)
   결과: 감지율이 무너져서 걷기 조건은 기본으로 쓰지 않음 (docs/agent_results.md 3~4절)
   지표는 12·13주차와 같음 + 강화·보호자 알림 횟수

실행: .venv\\Scripts\\python.exe src\\agent_eval.py   -> docs/agent_results.md
"""
import numpy as np
import pandas as pd

from agent import IDLE, WALK_LOOKBACK, Agent
from cadence import MIN_RHYTHM, SEG, rhythm
from calibrate import loso_scores, normal_false_alarms
from data import FS, ROOT, list_runs, load_run
from features import STEP, WIN
from train import NO_FOG, THRESHOLD, event_metrics, label_episodes

LOOKBACKS = [5, 10, 20, 30]
CONCLUSION = """- **가설:** 동결은 걷는 도중에 생기므로, 직전에 걷고 있지 않았다면(서 있음) 확률이 높아도 일부러 멈춘 것이다.
- **결과:** 오탐은 10분당 4.5 → 3.4회로 조금 줄었지만, 감지율이 80.6% → 49.8%로 무너졌다.
- **이유:** Daphnet의 동결 중 상당수는 리듬 있게 걷는 도중이 아니라 걷기 시작이나 방향 전환 때 생긴다
  (동결 시작 직전 5초 안에 걷고 있던 비율 58%). 반대로 오탐도 걷는 중에 많이 생긴다(60%).
  두 집단의 비율이 확인 범위(5~30초) 전체에서 비슷해서, 이 조건은 둘을 똑같이 거른다.
- **결정:** 걷기 조건은 기본으로 쓰지 않는다(`Agent(require_walking=False)`). 확률만 쓰는 v1과 비교해 상태 머신은
  감지율을 유지하고(80.6%), 감지 지연을 1.06 → 0.91초로 줄이며, 큐가 켜진 동안 확률이 잠깐 흔들려도 끊기지 않는다.
- **오탐은 다른 방법으로:** 13주차 개인 보정(임계값만 이동)에 이어 두 번째로 실패한 \"규칙으로 오탐 줄이기\"다.
  남은 방법은 사용자의 \"잘못된 큐\" 피드백을 모아 LLM 에이전트가 그 사람에게 오탐이 생기는 상황을 찾는 것이다.
  걷기 여부·리듬·걸음 속도는 계속 기록해서 그 분석에 쓴다.
- **한계:** 걷기 판단은 발목 자기상관 하나만 썼다. 방향 전환 감지(몸통 회전) 같은 다른 상황 신호는 시험하지 않았다."""
RHYTHM_PATH = ROOT / "data" / "processed" / "rhythm.csv.gz"


def window_rhythm() -> pd.DataFrame:
    """모든 윈도우의 걷기 리듬과 걸음 속도 (엣지 노드 on_sample과 같은 계산)."""
    if RHYTHM_PATH.exists():
        return pd.read_csv(RHYTHM_PATH)
    rows = []
    for s, r in list_runs():
        ankle = load_run(s, r)[["ankle_x", "ankle_y", "ankle_z"]].to_numpy(float)
        for end in range(WIN, len(ankle) + 1, STEP):                 # end = 윈도우 끝(미포함) 인덱스
            buf = ankle[max(0, end - SEG):end]
            stride, rh = rhythm(buf) if len(buf) >= 4 * FS else (np.nan, 0.0)
            walking = rh >= MIN_RHYTHM
            rows.append({"subject": s, "run": r, "t_end": (end - 1) / FS, "rhythm": round(rh, 3),
                         "walking": walking, "cadence": round(2 * 60 / stride) if walking else None})
        print(f"{s}{r} 걷기 리듬 계산 완료", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RHYTHM_PATH, index=False, compression="gzip")
    return out


def run_agent(w: pd.DataFrame, require_walking: bool, threshold: float = THRESHOLD) -> tuple[pd.Series, pd.DataFrame]:
    """윈도우마다 큐가 켜져 있는지(Series)와 모든 행동 목록을 돌려준다.
    threshold는 LLM 에이전트의 simulate_threshold가 다른 임계값을 미리 재생해 볼 때 바꾼다."""
    cue_on = pd.Series(False, index=w.index)
    actions = []
    for (s, r), g in w.groupby(["subject", "run"], sort=False):
        agent = Agent(log_path=None, require_walking=require_walking)
        for i, row in zip(g.index, g.itertuples()):
            ev = {"t": row.t_end, "prob_avg": row.avg, "threshold": threshold, "walking": bool(row.walking),
                  "rhythm": row.rhythm, "cadence": None if pd.isna(row.cadence) else row.cadence}
            for a in agent.step(ev):
                actions.append({"subject": s, "run": r, "truth_y": row.y, **a})
            cue_on[i] = agent.state != IDLE
    return cue_on, pd.DataFrame(actions)


def lookback_table(w: pd.DataFrame, episodes: pd.DataFrame, rh: pd.DataFrame) -> pd.DataFrame:
    """실제 동결 시작과 오탐(v1) 시작 직전에 걷고 있었는지 비교. 비율이 비슷하면 걷기 조건으로 둘을 구분할 수 없다."""
    walk_t = {k: g.loc[g.walking, "t_end"].to_numpy() for k, g in rh.groupby(["subject", "run"])}

    def since(s, r, t):
        prev = walk_t[(s, r)][walk_t[(s, r)] <= t]
        return t - prev.max() if len(prev) else np.inf

    ep_since = [since(e.subject, e.run, e.start) for e in episodes.itertuples()]
    fa_since = []
    for (s, r), g in w.groupby(["subject", "run"]):
        on = np.diff(np.concatenate([[0], (g["avg"] >= THRESHOLD).to_numpy(dtype=int)])) == 1
        eps = episodes[(episodes.subject == s) & (episodes.run == r)]
        for t in g["t_end"].to_numpy()[on]:
            if not ((eps.start - 2 <= t) & (t <= eps.end + 1)).any():
                fa_since.append(since(s, r, t))
    rows = {f"직전 {L}초 안에 걷기": [np.mean(np.array(ep_since) <= L), np.mean(np.array(fa_since) <= L)]
            for L in LOOKBACKS}
    return (pd.DataFrame(rows, index=[f"실제 동결 시작 ({len(ep_since)}건)", f"v1 오탐 시작 ({len(fa_since)}건)"])
            * 100).round(1)


def evaluate(w, episodes, alarm, actions=None) -> dict:
    w = w.assign(alarm=alarm.to_numpy())
    m = event_metrics(w, episodes)
    m["전체 정상 구간 오탐(10분당)"] = normal_false_alarms(w, episodes)
    if actions is not None and len(actions):
        nf_min = (w.subject.isin(NO_FOG)).sum() * 0.5 / 60
        nf = actions[actions.subject.isin(NO_FOG)]
        m["S04·S10 보호자 알림(10분당)"] = (nf.action == "notify_caregiver").sum() / nf_min * 10
        m["강화 횟수(전체)"] = int((actions.action == "cue_escalate").sum())
        m["보호자 알림(전체)"] = int((actions.action == "notify_caregiver").sum())
        m["걷기 조건으로 막은 판단(전체)"] = int(actions.reason.str.contains("걷지 않음").sum())
    return m


def main():
    w = loso_scores().merge(window_rhythm(), on=["subject", "run", "t_end"], how="left", validate="1:1")
    assert w["rhythm"].notna().all(), "걷기 리듬이 없는 윈도우가 있음"
    w = w.sort_values(["subject", "run", "t_end"]).reset_index(drop=True)
    episodes = label_episodes()

    rows = [{"방법": f"v1 알람 (확률 ≥ {THRESHOLD})", **evaluate(w, episodes, w["avg"] >= THRESHOLD)}]
    all_actions = {}
    for name, req in [("에이전트 (걷기 조건 없음)", False), (f"에이전트 (직전 {WALK_LOOKBACK:.0f}초 걷는 중)", True)]:
        cue_on, acts = run_agent(w, req)
        all_actions[name] = acts
        rows.append({"방법": name, **evaluate(w, episodes, cue_on, acts)})
        print(pd.DataFrame(rows).round(3).to_string(), flush=True)
    table = pd.DataFrame(rows).set_index("방법")

    # 걷기 조건을 켰을 때 환자별로 무엇을 막았나
    final = all_actions[rows[-1]["방법"]]
    blocked = final[final.reason.str.contains("걷지 않음")]
    per_subj = (pd.DataFrame({
        "큐 시작": final[final.action == "cue_start"].groupby("subject").size(),
        "걷기 조건으로 막음": blocked.groupby("subject").size(),
        "막은 판단 중 실제 동결 윈도우": blocked[blocked.truth_y == 1].groupby("subject").size()})
        .fillna(0).astype(int))
    per_subj.index.name = "환자"
    look = lookback_table(w, episodes, window_rhythm())
    print(look.to_string())
    write_report(table, per_subj, look)


def write_report(table: pd.DataFrame, per_subj: pd.DataFrame, look: pd.DataFrame):
    t = table.copy()
    t["에피소드 감지율"] = (t["에피소드 감지율"] * 100).round(1).astype(str) + "%"
    md = ["# 에이전트 평가: \"직전 5초 걷는 중\" 조건 (14주차)", "",
          "`src/agent_eval.py`로 생성. 질문: **확률만 보지 않고 \"걷다가 멈췄는가\"를 함께 보면 오탐이 줄어드는가?**", "",
          f"- 확률: 13주차와 같은 LOSO 확률 + 3윈도우 평균, 임계값 {THRESHOLD} 고정",
          "- 걷기 여부: 엣지 노드와 같은 계산(최근 10초 발목 가속도 자기상관 ≥ "
          f"{MIN_RHYTHM}). 윈도우를 시간 순서대로 `src/agent.py`의 상태 머신에 넣음",
          "- v1 알람: 확률이 임계값 이상인 윈도우마다 알람(12주차 방식). 에이전트: 큐 시작부터 회복(2윈도우 연속 아래)까지",
          "- 오탐 = 큐(알람)가 새로 켜진 횟수. 전체 정상 구간 오탐은 동결 시작 2초 전 ~ 끝 1초 후는 제외",
          "", "## 1. 결과", "", t.round(2).to_markdown(), "",
          "## 2. 환자별 (직전 5초 걷는 중 조건을 켰을 때)", "",
          "\"막은 판단\"은 확률이 높았지만 걷고 있지 않아서 행동하지 않은 0.5초 판단의 수(같은 상황이 이어지면 여러 번 셈).",
          "", per_subj.to_markdown(), "",
          "## 3. 왜 감지율이 떨어졌나: 동결 직전과 오탐 직전의 걷기 비율 (%)", "",
          look.to_markdown(), "",
          "두 줄이 거의 같다 → \"직전에 걷고 있었는가\"로는 진짜 동결과 오탐을 구분할 수 없다.", "",
          "## 4. 결론: 걷기 조건은 빼고, 상태 머신만 쓴다", "", CONCLUSION, ""]
    (ROOT / "docs" / "agent_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"-> {ROOT / 'docs' / 'agent_results.md'}")


if __name__ == "__main__":
    main()
