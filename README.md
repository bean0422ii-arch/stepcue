# StepCue

파킨슨 보행동결(FoG)을 감지해 리듬 신호를 주는 엣지 AI 에이전트 (IoT 수업 프로젝트, 2026 가을)

> 진행 중. 실행 방법(명령어 4개 이하)은 15주차에 채움. 진행 상황은 [todo.md](todo.md) 참고.

## 폴더 구조
```
data/raw/     Daphnet 원본 데이터 (git에 올리지 않음)
models/       학습된 모델 (.onnx)
src/          simulator.py, features.py, train.py, edge_node.py, agent.py
dashboard/    Streamlit 앱 (app.py)
scenarios/    테스트 시나리오 3개
logs/         decisions.jsonl (에이전트 결정 로그)
docs/         problem_card.md, design.md, ai_use.md, email_draft.md
assets/       수업 슬라이드 이미지
```

## 실행 방법
(15주차에 작성)
