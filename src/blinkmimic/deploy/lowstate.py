"""Only observation ingress used by the shared runner, for both simulated and hardware DDS."""
import numpy as np
def decode_snapshot(r):
    s={k:np.array(r[k][:29],copy=True) for k in ("q","dq","tau","temp","motorstate")}
    s.update({k:np.array(r[k],copy=True) for k in ("gyro","quat","acc")})
    s.update({k:float(r[k]) for k in ("t_rx","t_tick_change")})
    s.update(mode_machine=int(r["mode_machine"]),mode_pr=int(r["mode_pr"]),
             tick=int(r["tick"]),keys=int(r["wireless_remote"][2])|int(r["wireless_remote"][3])<<8)
    return s
