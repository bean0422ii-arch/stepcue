"""MQTT 공통 설정: 브로커 주소, 토픽 이름, 클라이언트 연결 함수 (시뮬레이터·엣지·에이전트·대시보드가 같이 사용)."""
import os
import time

import paho.mqtt.client as mqtt

HOST = "127.0.0.1"
PORT = int(os.environ.get("STEPCUE_MQTT_PORT", 1883))     # 기본 1883. 다른 포트를 쓰려면 환경변수로 지정

TOPIC_IMU = "stepcue/imu"          # 시뮬레이터 -> 엣지: 가속도 샘플 (64Hz)
TOPIC_EVENT = "stepcue/event"      # 엣지 -> 에이전트·대시보드: 0.5초마다 동결 확률 등
TOPIC_ACTION = "stepcue/action"    # 에이전트 -> 대시보드: 큐 시작/강화/중지, 보호자 알림


def connect(name: str, retries: int = 10) -> mqtt.Client:
    """브로커에 연결하고 네트워크 스레드를 시작한 클라이언트를 돌려준다.

    브로커가 아직 안 켜졌으면 1초 간격으로 다시 시도한다(여러 프로그램을 동시에 켤 때 순서 문제 방지).
    """
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"stepcue-{name}")
    for i in range(retries):
        try:
            client.connect(HOST, PORT, keepalive=30)
            break
        except (ConnectionRefusedError, OSError):
            if i == retries - 1:
                raise SystemExit(f"MQTT 브로커({HOST}:{PORT})에 연결할 수 없습니다.\n"
                                 "먼저 실행: .venv\\Scripts\\python.exe src\\broker.py")
            time.sleep(1)
    client.loop_start()
    return client
