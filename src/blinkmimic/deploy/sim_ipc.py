"""Typed simulation IPC. Runner does not import this module."""
import ctypes
from pathlib import Path
import numpy as np
STATE=np.dtype([("sample_wall","f8"),("sim_time","f8"),("tick","u8"),("keys","u8"),
    ("q","f8",29),("dq","f8",29),("tau","f8",29),("gyro","f8",3),("quat","f8",4),
    ("acc","f8",3),("temp","f8",(29,2)),("qpos","f8",36),("qvel","f8",35),
    ("support","u8"),("motor_fault","u8")],align=True)
CMD=np.dtype([("received","f8"),("seq","u8"),("q","f8",29),("dq","f8",29),
    ("kp","f8",29),("kd","f8",29),("tau","f8",29),("mode","u8",29),
    ("mode_machine","u8"),("mode_pr","u8")],align=True)
BRIDGE=Path(__file__).resolve().parents[3]/"scripts/deploy/bridge"
class SimIPC:
    def __init__(self,path,create=False):
        self.lib=ctypes.CDLL(str(BRIDGE/"libg1sim_client.so"))
        self.lib.fs_open.argtypes=[ctypes.c_char_p,ctypes.c_int];self.lib.fs_open.restype=ctypes.c_void_p
        self.lib.fs_close.argtypes=[ctypes.c_void_p]
        for n in ("fs_write_state","fs_read_state","fs_read_cmd","fs_counts"):
            getattr(self.lib,n).argtypes=[ctypes.c_void_p,ctypes.c_void_p]
        for n in ("fs_try_write_state","fs_try_read_cmd"):
            getattr(self.lib,n).argtypes=[ctypes.c_void_p,ctypes.c_void_p]
            getattr(self.lib,n).restype=ctypes.c_int
        if (self.lib.fs_state_size(),self.lib.fs_cmd_size())!=(STATE.itemsize,CMD.itemsize):
            raise RuntimeError("simulation IPC layout mismatch")
        self.h=self.lib.fs_open(str(path).encode(),int(create))
        if not self.h:raise RuntimeError("cannot open simulation IPC: "+str(path))
    def write(self,data):self.lib.fs_write_state(self.h,data.ctypes.data)
    def try_write(self,data):
        return bool(self.lib.fs_try_write_state(self.h,data.ctypes.data))
    def try_command(self,data):
        # On contention retain the previous received command; do not fabricate a new timestamp.
        return bool(self.lib.fs_try_read_cmd(self.h,data.ctypes.data))
    def state(self):
        x=np.zeros(1,STATE);self.lib.fs_read_state(self.h,x.ctypes.data);return x[0]
    def command(self):
        x=np.zeros(1,CMD);self.lib.fs_read_cmd(self.h,x.ctypes.data);return x[0]
    def counts(self):
        x=np.zeros(4,np.uint64);self.lib.fs_counts(self.h,x.ctypes.data);return x.tolist()
    def close(self):
        if self.h:self.lib.fs_close(self.h);self.h=None
