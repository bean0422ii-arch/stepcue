"""13주차 실험 B: 개인 보정. 환자마다 다른 임계값을 쓰면 오탐이 줄어드는가?

12주차 한계: 모든 환자에게 같은 임계값 0.8 -> 동결 없는 S04·S10 오탐 10분당 4.6회 (목표 1회 이하).

방법
1. 확률: 12주차와 같은 LOSO 확률 (그 환자를 빼고 학습한 모델의 출력) + 3윈도우 이동 평균
2. 보정: 환자마다 기록 앞부분 N분(2분, 5분)을 "보정 구간"으로 쓴다.
   보정 구간의 정상 윈도우(동결 레이블 아님) 확률의 상위 백분위(95%, 99%, 최댓값)를 개인 임계값으로 정한다.
   = "이 사람은 평소에도 확률이 여기까지 올라간다 -> 그보다 높을 때만 동결로 본다"
   범위는 0.5~0.95로 제한(너무 낮으면 오탐 폭발, 너무 높으면 감지 불가)
3. 평가: 보정 구간을 뺀 나머지로, 고정 0.8과 개인 임계값을 같은 데이터에서 비교
   - 에피소드 감지율, 감지 지연 중앙값, S04·S10 오탐(10분당) (12주차와 같은 지표)
   - 전체 환자 정상 구간 오탐(10분당): 동결 환자 8명도 포함. 동결 근처(시작 2초 전 ~ 끝 1초 후) 알람은 제외

규칙 후보는 미리 정해 두고 모두 표에 남긴다(가장 좋은 것만 골라 보고하면 결과가 부풀려지므로).

실행: .venv\\Scripts\\python.exe src\\calibrate.py
결과: docs/calibration_results.md, data/processed/loso_scores.csv.gz (LOSO 확률, 14주차 simulate_threshold에서 재사용)
"""
import numpy as np
import pandas as pd

from data import ROOT
from features import load_windows
from train import NO_FOG, THRESHOLD, RandomForestModel, event_metrics, label_episodes, loso, smooth

CAL_MINUTES = [2, 5]
RULES = {"95%": 0.95, "99%": 0.99, "최댓값": 1.0}
THR_RANGE = (0.5, 0.95)
PRE, POST = 2.0, 1.0           # 동결 시작 2초 전 ~ 끝 1초 후 알람은 "정상 구간 오탐"에서 제외
SCORES_PATH = ROOT / "data" / "processed" / "loso_scores.csv.gz"

INTERPRETATION = """\
**결론: 6가지 개인 규칙 모두 고정 0.8보다 나빴다.** S04·S10 오탐은 고정 0.8(10분당 약 4.9회)이 가장 적었고,
오탐을 줄인 규칙(5분 최댓값, 전체 정상 구간 오탐 2.2회)은 감지율이 17.8%로 무너졌다.

왜 안 됐나
1. **동결 없는 환자의 임계값이 오히려 내려갔다.** S04·S10의 개인 임계값은 0.5~0.79로 0.8보다 낮다.
   기록 앞부분(처음 몇 분)에는 확률이 낮게 나오고, 오탐을 일으키는 동작은 그 뒤에 나온다.
   → 앞부분 몇 분은 그 사람의 "평소 모습"을 대표하지 못한다.
2. **동결 환자는 정상 구간에서도 확률이 천장 근처까지 올라간다.** 정상 윈도우 최댓값이 0.84~0.89여서,
   이것을 임계값으로 쓰면 실제 동결(확률 0.9 안팎)과 구분할 여유가 거의 없다 → 감지율 급락.
3. 즉 오탐은 "사람마다 확률 기준선이 달라서"가 아니라 **특정 상황(동작)에서** 생긴다.
   임계값 하나를 사람마다 옮기는 방법으로는 상황을 구분할 수 없다.

다음 단계로 이어지는 점 (14주차)
- 임계값은 고정 0.8을 유지하고, 오탐은 **상황 조건**으로 줄인다: 에이전트의 "직전 5초 걷는 중" 조건.
- LLM 에이전트의 `run_calibration(minutes)` 도구는 빼고, "잘못된 큐" 피드백이 **어떤 상황에서** 생기는지
  (걷기 리듬, 걸음 속도, 직전 행동)를 찾는 쪽에 집중한다.
- 한계: 보정 규칙은 단순한 백분위만 시험했다. 보정 구간을 "걷기 과제"처럼 정해진 동작으로 녹화하면 결과가 다를 수 있다."""


def loso_scores() -> pd.DataFrame:
    """LOSO 확률을 계산(처음 한 번, 약 1분)하고 저장해 둔다."""
    if SCORES_PATH.exists():
        return pd.read_csv(SCORES_PATH)
    w = load_windows()
    w["score"] = loso(RandomForestModel, w).to_numpy()
    w["avg"] = smooth(w).to_numpy()
    out = w[["subject", "run", "t_end", "frac_fog", "y", "score", "avg"]]
    out.to_csv(SCORES_PATH, index=False, compression="gzip")
    return out


def split_calibration(w: pd.DataFrame, minutes: float) -> pd.Series:
    """환자마다 앞부분 minutes분(윈도우 0.5초 간격)을 True로 표시."""
    n = int(minutes * 60 * 2)
    return w.groupby("subject").cumcount() < n


def episodes_after(episodes: pd.DataFrame, w: pd.DataFrame, is_cal: pd.Series) -> pd.DataFrame:
    """보정 구간이 끝난 뒤 시작한 에피소드만 남긴다."""
    last = w[is_cal].groupby("subject").tail(1).set_index("subject")[["run", "t_end"]]
    keep = [ep.run > last.loc[ep.subject, "run"]
            or (ep.run == last.loc[ep.subject, "run"] and ep.start > last.loc[ep.subject, "t_end"])
            for ep in episodes.itertuples()]
    return episodes[keep]


def normal_false_alarms(w: pd.DataFrame, episodes: pd.DataFrame) -> float:
    """전체 환자의 정상 구간 오탐(10분당): 동결 근처가 아닌 곳에서 알람이 켜진 횟수."""
    count = 0
    for (s, r), g in w.groupby(["subject", "run"]):
        on = np.diff(np.concatenate([[0], g["alarm"].to_numpy(dtype=int)])) == 1
        eps = episodes[(episodes.subject == s) & (episodes.run == r)]
        for t in g["t_end"].to_numpy()[on]:
            if not ((eps.start - PRE <= t) & (t <= eps.end + POST)).any():
                count += 1
    minutes = (w["y"] == 0).sum() * 0.5 / 60
    return count / minutes * 10


def evaluate(w: pd.DataFrame, episodes: pd.DataFrame, thr: pd.Series) -> dict:
    w = w.assign(alarm=w["avg"].to_numpy() >= thr.to_numpy())
    m = event_metrics(w, episodes)
    m["전체 정상 구간 오탐(10분당)"] = normal_false_alarms(w, episodes)
    return m


def main():
    w = loso_scores()
    episodes = label_episodes()
    rows, thr_table = [], {}
    for minutes in CAL_MINUTES:
        is_cal = split_calibration(w, minutes)
        ev, eps = w[~is_cal].reset_index(drop=True), episodes_after(episodes, w, is_cal)
        cal_normal = w[is_cal & (w["y"] == 0)]

        fixed = pd.Series(THRESHOLD, index=ev.index)
        rows.append({"보정 길이": f"{minutes}분", "방법": f"고정 {THRESHOLD}", **evaluate(ev, eps, fixed),
                     "평가 에피소드": len(eps)})
        for name, q in RULES.items():
            personal = cal_normal.groupby("subject")["avg"].quantile(q).clip(*THR_RANGE)
            thr_table[f"{minutes}분 {name}"] = personal
            rows.append({"보정 길이": f"{minutes}분", "방법": f"개인 {name}",
                         **evaluate(ev, eps, ev["subject"].map(personal)), "평가 에피소드": len(eps)})
        print(pd.DataFrame(rows[-4:]).round(3).to_string())

    table = pd.DataFrame(rows).set_index(["보정 길이", "방법"])
    thr_df = pd.DataFrame(thr_table).round(2)
    thr_df.index.name = "환자"
    write_report(table, thr_df)


def write_report(table: pd.DataFrame, thr_df: pd.DataFrame):
    t = table.copy()
    t["에피소드 감지율"] = (t["에피소드 감지율"] * 100).round(1).astype(str) + "%"
    md = ["# 개인 보정 실험 결과 (13주차, 실험 B)", "",
          "`src/calibrate.py`로 생성. 질문: **환자마다 다른 임계값을 쓰면 오탐이 줄어드는가?**", "",
          "- 확률: 12주차와 같은 LOSO 확률(그 환자를 빼고 학습한 모델) + 3윈도우 평균",
          "- 개인 임계값: 기록 앞부분 N분의 정상 윈도우 확률 중 상위 백분위 (범위 0.5~0.95로 제한)",
          "- 평가: 보정 구간을 뺀 나머지 데이터. 고정 0.8도 같은 데이터로 다시 계산(12주차 표와 숫자가 조금 다름)",
          f"- 전체 정상 구간 오탐: 동결 환자 포함 10명. 동결 시작 {PRE:.0f}초 전 ~ 끝 {POST:.0f}초 후 알람은 제외",
          "- 규칙 후보(95%, 99%, 최댓값)는 실험 전에 정했고 모두 표에 남김", "",
          "## 1. 결과", "", t.round(2).to_markdown(), "",
          "## 2. 환자별 개인 임계값", "", thr_df.to_markdown(), "",
          "## 3. 해석: 효과 없음 → 개인 보정은 접는다", "", INTERPRETATION, ""]
    (ROOT / "docs" / "calibration_results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
