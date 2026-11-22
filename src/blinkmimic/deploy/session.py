"""Session naming and atomic UI status; no acceptance or arming gates."""
import hashlib
import json
import os
import time
from pathlib import Path
DEFAULT_PROFILE=dict(imu_frame="pelvis_rigid",quaternion_order="wxyz",gyro_units="rad/s",
                     R_pelvis_from_imu_wxyz=[1,0,0,0],
                     provenance="SuperMimic pelvis-IMU reference; not calibrated on this G1")
def new_session(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    stamp=time.strftime("%Y%m%d_%H%M%S")+f"_{os.getpid()}"
    path=root/stamp;path.mkdir()
    return path
def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False)+"\n");tmp.replace(path)


def buttons_address(session):
    """Linux AF_UNIX sun_path has 108 bytes including its terminator.
    Keep short-path sessions compatible; long paths use a kernel-owned abstract
    address (released automatically when physics exits), never a shared /tmp file.
    """
    path=str(Path(session).resolve()/"buttons.sock")
    if len(os.fsencode(path))<=107:
        return path
    key=hashlib.sha256(os.fsencode(path)).hexdigest()
    return "\0blinkmimic-buttons-"+key
