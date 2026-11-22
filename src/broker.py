"""13주차: MQTT 브로커 (파이썬 패키지 amqtt 사용).

Mosquitto를 따로 설치하지 않아도 되도록 파이썬으로 브로커를 띄운다. (README 명령어 수를 줄이기 위함)
노트북 안에서만 쓰므로 127.0.0.1(내 컴퓨터)에서만 접속을 받는다.
이미 다른 브로커(예: Mosquitto 서비스)가 같은 포트에서 실행 중이면 그것을 그대로 쓰고 종료한다.

실행: .venv\\Scripts\\python.exe src\\broker.py      (Ctrl+C로 종료)
"""
import asyncio
import logging
import socket

from amqtt.broker import Broker

from mqtt_common import HOST, PORT

CONFIG = {
    "listeners": {"default": {"type": "tcp", "bind": f"{HOST}:{PORT}"}},
    "plugins": {"amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True}},
}


def port_in_use() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


async def run():
    broker = Broker(CONFIG)
    await broker.start()
    print(f"MQTT 브로커 실행 중: {HOST}:{PORT}  (Ctrl+C로 종료)", flush=True)
    try:
        await asyncio.Event().wait()        # 종료할 때까지 대기
    finally:
        await broker.shutdown()


def main():
    if port_in_use():
        print(f"{HOST}:{PORT}에 이미 브로커가 실행 중입니다(예: Mosquitto). 그 브로커를 그대로 사용하세요.")
        return
    logging.basicConfig(level=logging.WARNING)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("브로커 종료")


if __name__ == "__main__":
    main()
