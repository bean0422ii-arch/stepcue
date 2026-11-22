"""13주차: 모델을 ONNX로 변환하고, 엣지 지표(모델 크기, 윈도우당 처리 시간)를 측정한다.

ONNX: 모델을 "어떤 언어·기기에서도 실행할 수 있는 공통 형식"으로 저장한 파일.
  - pickle(rf.pkl)은 파이썬 + 같은 버전의 scikit-learn이 있어야만 열 수 있다.
  - ONNX(rf.onnx)는 onnxruntime만 있으면 C/C++, 안드로이드, 라즈베리파이 등에서도 실행 가능 -> 실제 웨어러블로 옮기기 쉬움.

확인하는 것
1. 정확도: 같은 입력에 대해 pickle 모델과 ONNX 모델의 확률이 같은가 (ONNX는 float32라서 아주 작은 차이는 생김)
2. 크기: KB (목표 500KB 미만)
3. 속도: 윈도우 1개 처리 시간 ms (목표 10ms 미만). 특징 추출 + 추론을 나눠서 측정

실행: .venv\\Scripts\\python.exe src\\export_onnx.py
결과: models/rf.onnx, docs/edge_metrics.md
"""
import pickle
import time

import numpy as np
from skl2onnx import to_onnx

from data import FS, ROOT, load_run
from features import FEATURE_NAMES, WIN, extract, load_windows
from onnx_model import ONNX_PATH, load_session
from onnx_model import predict_prob as onnx_prob
from train import THRESHOLD

PKL_PATH = ROOT / "models" / "rf.pkl"
N_TIMING = 1000                 # 윈도우 1000개를 하나씩 처리해서 시간 측정


def timing(fn, n: int) -> tuple[float, float]:
    """fn을 n번 실행 -> (중앙값 ms, 상위 95% ms)"""
    ts = []
    for i in range(n):
        t = time.perf_counter()
        fn(i)
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    with open(PKL_PATH, "rb") as f:
        clf = pickle.load(f)
    clf.n_jobs = 1

    # 1. 변환 (zipmap=False: 확률을 딕셔너리가 아니라 배열로 받기)
    onx = to_onnx(clf, np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32),
                  options={id(clf): {"zipmap": False}}, target_opset=17)
    ONNX_PATH.write_bytes(onx.SerializeToString())
    sess = load_session()
    input_name = sess.get_inputs()[0].name
    assert input_name == "X", f"입력 이름이 예상과 다름: {input_name}"

    # 2. 정확도: 전체 윈도우(약 4만 개)에서 두 모델 확률 비교
    w = load_windows()
    X = w[FEATURE_NAMES].to_numpy()
    p_pkl = clf.predict_proba(X)[:, 1]
    p_onnx = onnx_prob(sess, X)
    diff = np.abs(p_pkl - p_onnx)
    agree = ((p_pkl >= THRESHOLD) == (p_onnx >= THRESHOLD)).mean()

    # 3. 크기
    kb_pkl, kb_onnx = PKL_PATH.stat().st_size / 1024, ONNX_PATH.stat().st_size / 1024

    # 4. 속도: 실제 데이터에서 2초 윈도우를 0.5초 간격으로 잘라 하나씩 처리
    acc = load_run("S01", "R01").iloc[:, 1:10].to_numpy(dtype=float)
    wins = [acc[i * (FS // 2): i * (FS // 2) + WIN] for i in range(N_TIMING)]
    feats = [extract(x) for x in wins]
    t_feat = timing(lambda i: extract(wins[i]), N_TIMING)
    t_pkl = timing(lambda i: clf.predict_proba(feats[i]), N_TIMING)
    t_onnx = timing(lambda i: onnx_prob(sess, feats[i]), N_TIMING)

    rows = [("특징 추출 (FFT, 10개)", *t_feat), ("추론: pickle (scikit-learn)", *t_pkl),
            ("추론: ONNX (onnxruntime)", *t_onnx),
            ("합계: 특징 + ONNX 추론", t_feat[0] + t_onnx[0], t_feat[1] + t_onnx[1])]
    for r in rows:
        print(f"{r[0]:<28} 중앙값 {r[1]:6.3f}ms  95% {r[2]:6.3f}ms")
    print(f"크기: pickle {kb_pkl:.0f}KB, ONNX {kb_onnx:.0f}KB")
    print(f"확률 차이: 최대 {diff.max():.2e}, 평균 {diff.mean():.2e} / 임계값 {THRESHOLD} 판단 일치 {agree:.4%}")

    md = ["# 엣지 지표 (13주차)", "",
          "`src/export_onnx.py`로 생성. 엣지 노드와 같은 조건(윈도우 1개씩, 1스레드)으로 측정. 노트북 CPU 기준.", "",
          "## 1. ONNX 변환", "",
          "| 항목 | pickle (rf.pkl) | ONNX (rf.onnx) | 목표 |", "|---|---|---|---|",
          f"| 모델 크기 | {kb_pkl:.0f} KB | **{kb_onnx:.0f} KB** | 500 KB 미만 |",
          f"| 윈도우당 추론 시간 (중앙값) | {t_pkl[0]:.2f} ms | **{t_onnx[0]:.3f} ms** | 10 ms 미만 |",
          f"| 윈도우당 추론 시간 (95%) | {t_pkl[1]:.2f} ms | {t_onnx[1]:.3f} ms | |", "",
          f"- 정확도 확인: 윈도우 {len(X):,}개에서 두 모델의 동결 확률 차이 최대 {diff.max():.1e} "
          f"(ONNX는 float32 사용). 임계값 {THRESHOLD} 기준 판단 일치율 {agree:.3%}",
          "- ONNX는 onnxruntime만 있으면 파이썬·scikit-learn 없이도 실행할 수 있어서, 실제 웨어러블(C/C++, 안드로이드)로 옮기기 쉽다", "",
          "## 2. 윈도우 1개 처리 시간 (0.5초마다 1번)", "",
          "| 단계 | 중앙값 (ms) | 95% (ms) |", "|---|---|---|"]
    md += [f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} |" for r in rows]
    md += ["", f"- 측정: S01 데이터에서 2초 윈도우 {N_TIMING}개를 하나씩 처리",
           "- 0.5초(500ms)마다 한 번 처리하므로 CPU를 거의 쓰지 않는다. 엣지 노드는 여기에 걸음 리듬 계산(10초 자기상관)이 추가됨", ""]
    (ROOT / "docs" / "edge_metrics.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
