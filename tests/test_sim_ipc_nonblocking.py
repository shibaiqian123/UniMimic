import ctypes
import subprocess
import time
import numpy as np
from blinkmimic.deploy.sim_ipc import SimIPC,STATE,CMD,BRIDGE

def test_physics_does_not_wait_for_other_ipc_users(tmp_path):
    source=tmp_path/"hold.cpp"
    source.write_text('#include "g1_sim_shm.h"\nextern "C" void hold(void* p,int cmd){auto* s=(SimShared*)p;lock_sim(cmd? &s->cmd_mu:&s->state_mu);}\nextern "C" void release(void* p,int cmd){auto* s=(SimShared*)p;pthread_mutex_unlock(cmd? &s->cmd_mu:&s->state_mu);}')
    libpath=tmp_path/"hold.so"
    subprocess.run(["g++","-shared","-fPIC","-I",str(BRIDGE),str(source),"-o",str(libpath),"-lpthread"],check=True)
    helper=ctypes.CDLL(str(libpath))
    for n in ("hold","release"):getattr(helper,n).argtypes=[ctypes.c_void_p,ctypes.c_int]
    ipc=SimIPC(tmp_path/"test.mmap",create=True)
    state=np.zeros(1,STATE);state["tick"]=17
    command=np.zeros(1,CMD);command["seq"]=91
    try:
        for mode,call,data in ((0,ipc.try_write,state),(1,ipc.try_command,command)):
            helper.hold(ipc.h,mode)
            try:
                begin=time.monotonic()
                assert not call(data)
                assert time.monotonic()-begin<.02
            finally:helper.release(ipc.h,mode)
        assert command["seq"][0]==91
        assert ipc.try_write(state)
        assert ipc.state()["tick"]==17
        assert ipc.try_command(command)
        assert command["seq"][0]==0
    finally:ipc.close()
