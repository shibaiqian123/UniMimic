"""One pulse adapter shared by window, terminal and automatic simulation input."""
from .controller import PREP,ARM,PLAY,STOP
MAP={"1":PREP,"2":ARM,"3":PLAY,"D":STOP}
class ButtonPulses:
    def __init__(self):
        self.keys=0;self.until=0.;self.release_until=0.;self.label=""
    def send(self,key,now):
        key=key.upper()
        if key not in MAP:return False
        if key=="D":
            self.keys=STOP;self.label=key;self.until=now+.15;self.release_until=self.until+.1
            return True
        self.sample(now)
        if now<self.release_until:return False
        self.keys=MAP[key];self.label=key;self.until=now+.15;self.release_until=self.until+.1
        return True
    def release(self,now):
        if self.keys==STOP and now<self.until:return # do not cancel an already-requested stop
        self.keys=0;self.label="";self.until=now;self.release_until=max(self.release_until,now+.1)
    def sample(self,now):
        if now>=self.until:self.keys=0;self.label=""
        return self.keys

def key_event(key):
    key=key.upper()
    if key not in ("1","2","3","D","B"):raise ValueError("unknown simulation key")
    return {"type":"key","key":key}

class WindowInput:
    """GLFW event adapter; PRESS=1, REPEAT=2. No controller access."""
    def __init__(self,emit):self.emit=emit
    def key(self,key,action):
        if action!=1 or key not in (49,50,51,68,66):return False
        self.emit(key_event(chr(key)));return True
    def focus(self,focused):
        if not focused:self.emit({"type":"release"})
