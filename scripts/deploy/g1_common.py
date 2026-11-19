"""Minimal G1 state wire contract. Adapted from SuperMimic (see bridge/LICENSE)."""
from dataclasses import dataclass
import numpy as np
NUM_JOINTS=29
@dataclass
class G1State:
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    temp: np.ndarray
    gyro: np.ndarray
    quat: np.ndarray
    acc: np.ndarray
    rpy: np.ndarray
    tick: int
    mode_machine: int
    mode_pr: int
    keys: int
    motorstate: np.ndarray
def init_dds(domain,iface):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(domain,iface)

BUTTON_BITS={"R1":0,"L2":5,"A":8,"B":9,"X":10,"up":12}
def combo_mask(*names):
    return sum(1<<BUTTON_BITS[n] for n in names)
