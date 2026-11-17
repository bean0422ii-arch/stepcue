"""12주차: 베이스라인(Freeze Index 임계값)과 RandomForest를 환자 단위 교차검증(LOSO)으로 비교.

LOSO(Leave-One-Subject-Out): 환자 1명을 빼고 나머지로 학습/임계값 결정 -> 뺀 환자로 평가. 10번 반복.
"처음 보는 환자에게도 통하는가"를 확인하는 방법이다.

평가 지표
- 윈도우 단위: 민감도(동결을 동결로), 특이도(정상을 정상으로), F1
- 에피소드 단위 (problem card 성공 기준): 동결 에피소드 감지율, 감지 지연, S04/S10 오탐(10분당)
  에이전트처럼 확률을 최근 3윈도우 평균으로 부드럽게 한 뒤 계산한다.

실행: .venv\\Scripts\\python.exe src\\train.py
결과: docs/model_results.md, docs/figures/model_compare.png, models/rf.pkl (전체 데이터로 다시 학습한 최종 모델)
"""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data import FS, ROOT, load_run
from features import FEATURE_NAMES, load_windows

SMOOTH = 3                  # 에이전트와 같은 3윈도우 이동 평균
DETECT_GRACE = 1.0          # 에피소드 끝난 뒤 1초 안에 잡아도 감지로 인정
NO_FOG = ["S04", "S10"]
THRESHOLD = 0.8            # RF 동결 판단 임계값 (임계값별 결과표를 보고 감지율 80%를 지키는 가장 높은 값으로 선택)
SWEEP = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85]
RF_PARAMS = dict(n_estimators=20, max_depth=6, min_samples_leaf=20,   # 모델 < 500KB 목표 때문에 작게
                 class_weight="balanced", random_state=0, n_jobs=-1)
MODEL_PATH = ROOT / "models" / "rf.pkl"


# ---------- 베이스라인: Freeze Index 임계값 + 움직임 임계값 ----------
class FreezeIndexBaseline:
    """fi_ankle > fi_thr 이고 pow_ankle > pow_thr(서 있지 않음)면 동결. 임계값은 학습 환자에서 F1 최대로 고른다."""

    def fit(self, X: pd.DataFrame, y):
        best = (-1, 0, 0)
        for p in np.quantile(X["pow_ankle"], np.linspace(0, 0.6, 13)):
            moving = X["pow_ankle"].to_numpy() > p
            for f in np.quantile(X["fi_ankle"], np.linspace(0.5, 0.98, 25)):
                pred = moving & (X["fi_ankle"].to_numpy() > f)
                best = max(best, (f1(y, pred), f, p))
        _, self.fi_thr, self.pow_thr = best
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """0~1 점수로 바꿔서 모델 확률처럼 다룬다(임계값 위치 = 0.5)."""
        moving = X["pow_ankle"].to_numpy() > self.pow_thr
        return np.where(moving, 1 / (1 + np.exp(-4 * (X["fi_ankle"].to_numpy() - self.fi_thr))), 0.0)


class RandomForestModel:
    def fit(self, X, y):
        self.clf = RandomForestClassifier(**RF_PARAMS).fit(X[FEATURE_NAMES].to_numpy(), y)
        return self

    def score(self, X):
        return self.clf.predict_proba(X[FEATURE_NAMES].to_numpy())[:, 1]


# ---------- 지표 ----------
def f1(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp = (y & pred).sum()
    return 2 * tp / max(y.sum() + pred.sum(), 1)


def window_metrics(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return {"민감도": (y & pred).sum() / y.sum() if y.any() else np.nan,   # 동결 없는 환자는 계산 불가
            "특이도": (~y & ~pred).sum() / max((~y).sum(), 1),
            "F1": f1(y, pred)}


def smooth(w: pd.DataFrame, col="score") -> pd.Series:
    return w.groupby(["subject", "run"])[col].transform(lambda s: s.rolling(SMOOTH, min_periods=1).mean())


def label_episodes() -> pd.DataFrame:
    """원본 레이블에서 동결 에피소드(시작·끝 초)를 뽑는다."""
    rows = []
    for (s, r), _ in load_windows().groupby(["subject", "run"]):
        lab = (load_run(s, r)["label"].to_numpy() == 2).astype(int)
        e = np.diff(np.concatenate([[0], lab, [0]]))
        for a, b in zip(np.where(e == 1)[0], np.where(e == -1)[0]):
            rows.append({"subject": s, "run": r, "start": a / FS, "end": b / FS})
    return pd.DataFrame(rows)


def event_metrics(w: pd.DataFrame, episodes: pd.DataFrame) -> dict:
    """부드럽게 한 예측(alarm)으로 에피소드 감지율, 감지 지연, 오탐 횟수 계산."""
    detected, delays = 0, []
    for ep in episodes.itertuples():
        g = w[(w.subject == ep.subject) & (w.run == ep.run)]
        hit = g[g.alarm & (g.t_end >= ep.start) & (g.t_end <= ep.end + DETECT_GRACE)]
        if len(hit):
            detected += 1
            delays.append(hit.t_end.iloc[0] - ep.start)
    # 오탐: 동결이 없는 환자에서 알람이 켜진 횟수(꺼짐 -> 켜짐)
    nf = w[w.subject.isin(NO_FOG)]
    starts = nf.groupby(["subject", "run"])["alarm"].apply(
        lambda a: int(np.diff(np.concatenate([[0], a.to_numpy(dtype=int)])).clip(0).sum())).sum()
    minutes = len(nf) * 0.5 / 60
    return {"에피소드 감지율": detected / len(episodes),
            "감지 지연 중앙값(초)": float(np.median(delays)) if delays else np.nan,
            "S04·S10 오탐(10분당)": starts / minutes * 10}


# ---------- LOSO ----------
def loso(model_cls, w: pd.DataFrame) -> pd.Series:
    scores = pd.Series(np.nan, index=w.index)
    for subj in sorted(w.subject.unique()):
        test = w.subject == subj
        m = model_cls().fit(w[~test], w.loc[~test, "y"].to_numpy())
        scores[test] = m.score(w[test])
    return scores


def evaluate(name, w, episodes, scores, thr=0.5):
    w = w.assign(score=scores.to_numpy())
    w["alarm"] = smooth(w) >= thr
    pred = w["score"] >= thr
    row = {"방법": name, **window_metrics(w.y, pred), **event_metrics(w, episodes)}
    per_subj = (w.assign(pred=pred).groupby("subject")
                .apply(lambda g: pd.Series(window_metrics(g.y, g.pred)), include_groups=False))
    return row, per_subj


def main():
    w = load_windows()
    episodes = label_episodes()
    print(f"윈도우 {len(w)}개, 에피소드 {len(episodes)}개")

    rf_scores = loso(RandomForestModel, w)
    results, per = [], {}
    for name, scores, thr in [("베이스라인 (Freeze Index 임계값)", loso(FreezeIndexBaseline, w), 0.5),
                              ("RandomForest, 임계값 0.5", rf_scores, 0.5),
                              (f"RandomForest, 임계값 {THRESHOLD} (선택)", rf_scores, THRESHOLD)]:
        row, ps = evaluate(name, w, episodes, scores, thr)
        results.append(row)
        if thr != 0.5 or "베이스라인" in name:
            per[name] = ps
        print(row)
    table = pd.DataFrame(results).set_index("방법")
    sweep = pd.DataFrame([evaluate(f"{t}", w, episodes, rf_scores, t)[0] for t in SWEEP]).set_index("방법")
    sweep.index.name = "RF 임계값"

    # 최종 모델: 전체 환자로 다시 학습해서 저장 (13주차에 ONNX로 변환)
    final = RandomForestModel().fit(w, w.y.to_numpy())
    final.clf.n_jobs = 1    # 실시간에는 윈도우 1개씩 예측 -> 병렬 처리하면 오히려 느림(19ms -> 1.5ms)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final.clf, f)
    size_kb = MODEL_PATH.stat().st_size / 1024
    importances = pd.Series(final.clf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)

    # 그림: 환자별 F1 비교
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    f1s = pd.DataFrame({k.split(" (")[0].split(",")[0]: v["F1"] for k, v in per.items()}).drop(NO_FOG, errors="ignore")
    ax = f1s.plot.bar(figsize=(8, 4), color=["#9db4c0", "#d1495b"])
    ax.set_title("환자별 F1 (LOSO, 동결 없는 S04·S10 제외)")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(ROOT / "docs" / "figures" / "model_compare.png", dpi=120)
    plt.close()

    def pretty(t):
        t = t.copy()
        for c in ["민감도", "특이도", "F1", "에피소드 감지율"]:
            t[c] = (t[c] * 100).round(1).astype(str) + "%"
        return t.round(2)
    fmt = pretty(table)
    md = ["# 모델 성능 비교 (12주차)", "",
          "`src/train.py`로 생성. 환자 단위 교차검증(LOSO, 10명). 윈도우 2초 / 0.5초 간격. "
          "실험 외(레이블 0) 구간이 섞인 윈도우는 제외.", "",
          "## 1. 전체 결과", "", fmt.to_markdown(), "",
          "- 윈도우 단위: 확률(점수) ≥ 임계값이면 동결로 판단 (베이스라인은 학습 환자에서 F1 최대가 되는 Freeze Index·파워 임계값)",
          f"- 에피소드 단위: 에이전트처럼 최근 {SMOOTH}윈도우 평균 ≥ 임계값이면 알람. "
          f"에피소드 시작부터 끝난 뒤 {DETECT_GRACE:.0f}초 사이에 알람이 켜지면 감지",
          "- 감지 지연: 레이블상 동결 시작 → 첫 알람(윈도우 끝 시각) 시간",
          "- 오탐: 동결이 없는 S04·S10에서 알람이 켜진 횟수. 14주차 에이전트의 \"직전에 걷고 있었는가\" 조건은 아직 미적용",
          "", "## 2. 임계값에 따른 변화 (RandomForest)", "",
          "임계값을 올리면 오탐은 줄지만 감지율이 떨어지고 감지가 늦어진다. "
          "감지 모델만으로는 감지율 80%와 오탐 10분당 1회 이하를 동시에 만족하지 못한다.", "",
          pretty(sweep).to_markdown(), "",
          f"→ 감지율 80% 이상을 지키는 가장 높은 값인 **{THRESHOLD}** 선택. "
          "(LOSO 결과를 보고 고른 값이라 실제 성능은 이보다 조금 낮을 수 있음.) "
          "남은 오탐은 14주차 에이전트 조건(직전 걷기 여부, 지속 시간)으로 줄인다.", "",
          "## 3. 환자별 결과 (윈도우 단위, %)", "",
          "S04·S10은 동결이 없어서 민감도·F1 계산 불가(NaN 또는 0).", ""]
    for k, v in per.items():
        md += [f"### {k}", "", (v * 100).round(1).to_markdown(), ""]
    md += ["![환자별 F1](figures/model_compare.png)", "",
           "## 4. 최종 모델", "",
           f"- 전체 환자로 다시 학습: `models/rf.pkl` ({size_kb:.0f} KB, pickle 기준. 13주차에 ONNX로 변환해 다시 측정)",
           f"- 설정: {RF_PARAMS}", "",
           "특징 중요도:", "", (importances.round(3).to_frame("중요도")).to_markdown(), ""]
    (ROOT / "docs" / "model_results.md").write_text("\n".join(md), encoding="utf-8")
    print(table.round(3).to_string())
    print(f"모델 크기 {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
