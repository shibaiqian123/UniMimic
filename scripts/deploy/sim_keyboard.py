"""Local keyboard client for a final SIMULATION session. Does not create DDS publishers."""
import argparse,json,socket,sys,termios,tty,time,select
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from blinkmimic.deploy.buttons import key_event
from blinkmimic.deploy.session import buttons_address

def send(session,event):
    session=Path(session)
    meta=json.loads((session/"session.json").read_text())
    if meta.get("kind")!="blink_final_sim" or meta.get("domain") not in (1,2):raise ValueError("simulation session required")
    with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as s:
        s.sendto(json.dumps(event).encode(),buttons_address(session))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--session",required=True,type=Path)
    ap.add_argument("--key",choices=["1","2","3","D","B"]);a=ap.parse_args()
    if a.key:send(a.session,key_event(a.key));return
    if not sys.stdin.isatty():ap.error("interactive TTY required, or use --key")
    print("1 prepare | 2 standby | 3 play | D stop | B support | Ctrl-C stop and exit",flush=True)
    old=termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        last=None
        while True:
            ready,_,_=select.select([sys.stdin],[],[],.1)
            if ready:
                key=sys.stdin.read(1).upper()
                if key in ("1","2","3","D","B"):send(a.session,key_event(key))
            try:
                status=json.loads((a.session/"runner_status.json").read_text())
                physics=json.loads((a.session/"physics_status.json").read_text())
                value=(status.get("state"),status.get("reject_reason") or status.get("reason"),physics.get("support"))
                if value!=last:print(f"\r\n{value[0]} | support={value[2]} | {value[1] or ''}",flush=True);last=value
            except (OSError,ValueError):pass
            if (a.session/"report.json").exists():print("\r\nSession finished",flush=True);break
    except KeyboardInterrupt:
        send(a.session,key_event("D"));time.sleep(.2)
    finally:
        termios.tcsetattr(sys.stdin,termios.TCSADRAIN,old)
        try:send(a.session,dict(type="release"))
        except OSError:pass
if __name__=="__main__":main()
