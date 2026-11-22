"""ONNX 모델 불러오기와 추론 (엣지 노드와 export_onnx.py가 같이 사용).

scikit-learn 없이 onnxruntime만으로 실행된다 -> 실제 웨어러블 기기로 옮기기 쉬움.
"""
import numpy as np
import onnxruntime as ort

from data import ROOT

ONNX_PATH = ROOT / "models" / "rf.onnx"


def load_session() -> ort.InferenceSession:
    """1스레드로 실행 (윈도우 1개씩 처리할 때는 병렬 준비 비용이 더 큼)."""
    if not ONNX_PATH.exists():
        raise SystemExit(f"ONNX 모델이 없습니다: {ONNX_PATH}\n"
                         "먼저 실행: .venv\\Scripts\\python.exe src\\export_onnx.py")
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 1
    opt.inter_op_num_threads = 1
    return ort.InferenceSession(str(ONNX_PATH), opt, providers=["CPUExecutionProvider"])


def predict_prob(sess: ort.InferenceSession, feats: np.ndarray) -> np.ndarray:
    """특징 (n, 10) -> 동결 확률 (n,)"""
    out = sess.run(None, {"X": np.asarray(feats, dtype=np.float32)})
    return out[1][:, 1]          # out[0] = 예측 클래스, out[1] = 클래스별 확률
