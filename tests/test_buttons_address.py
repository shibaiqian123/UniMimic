import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from blinkmimic.deploy.session import buttons_address
from blinkmimic.deploy.final_session import send

def test_short_address_compatible():
    assert buttons_address("/tmp/blink-test")=="/tmp/blink-test/buttons.sock"

def test_long_and_utf8_addresses_are_bounded_and_distinct(tmp_path):
    a=buttons_address(tmp_path/("组合动作"*30))
    b=buttons_address(tmp_path/("组合动作"*30+"2"))
    assert a.startswith("\0") and a!=b
    assert len(os.fsencode(a))<=108

def test_long_session_supervisor_and_keyboard_deliver_same_events(tmp_path):
    session=tmp_path/("shanshenbu_combo4_"*8)
    session.mkdir()
    (session/"session.json").write_text(json.dumps(dict(kind="blink_final_sim",domain=2)))
    address=buttons_address(session)
    assert address.startswith("\0")
    receiver=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM)
    receiver.settimeout(2)
    try:
        receiver.bind(address)
        event=dict(type="key",key="3")
        send(session,event)
        assert json.loads(receiver.recv(2048))==event
        root=Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable,str(root/"scripts/deploy/sim_keyboard.py"),
                        "--session",str(session),"--key","3"],check=True)
        assert json.loads(receiver.recv(2048))==event
    finally:receiver.close()
    # No stale filesystem socket or cleanup step after process exit.
    with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as receiver:
        receiver.bind(address)
