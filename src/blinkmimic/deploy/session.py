"""Session naming and atomic UI status; no acceptance or arming gates."""
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
