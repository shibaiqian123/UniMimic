"""SONIC topic/header/payload protocol; independently implemented receiver.
Reference: GR00T-WholeBodyControl zmq_planner_sender.py (Apache-2.0).
"""
import json
import numpy as np

DTYPES={"u8":"u1","bool":"?","i32":"<i4","i64":"<i8","f32":"<f4","f64":"<f8"}
TOPICS=(b"command",b"planner",b"pose")

def unpack(packet):
    topic=next((t for t in TOPICS if packet.startswith(t)),None)
    if topic is None:raise ValueError("unknown SONIC topic")
    # Release variants use 1024 or 1280 padded JSON headers. Detect by exact payload size.
    for size in (1280,1024):
        try:
            header=packet[len(topic):len(topic)+size]
            h=json.loads(header.rstrip(b"\0"))
            if h.get("endian","le")!="le":raise ValueError("unsupported endian")
            offset=len(topic)+size; result={}
            for f in h["fields"]:
                if f["name"] in result:raise ValueError("duplicate field")
                dtype=np.dtype(DTYPES[f["dtype"]]);shape=tuple(f["shape"])
                if any(not isinstance(d,int) or d<0 for d in shape):raise ValueError("invalid shape")
                count=int(np.prod(shape));length=count*dtype.itemsize
                if offset+length>len(packet):raise ValueError("truncated payload")
                a=np.frombuffer(packet,dtype=dtype,count=count,offset=offset).reshape(shape).copy()
                if not np.isfinite(a).all():raise ValueError("nonfinite packet")
                result[f["name"]]=a;offset+=length
            if offset!=len(packet):continue
            return topic.decode(),result
        except (ValueError,KeyError,TypeError,UnicodeDecodeError):continue
    raise ValueError("invalid SONIC packet")

# For replay/tests: matches SONIC field ordering and padding, no pickle.
def pack(topic,values,size=1280):
    reverse={np.dtype(v):k for k,v in DTYPES.items()}
    fields=[];parts=[]
    for name,value in values.items():
        a=np.ascontiguousarray(value)
        fields.append(dict(name=name,dtype=reverse[a.dtype],shape=list(a.shape)))
        parts.append(a.tobytes())
    h=json.dumps(dict(v=3,endian="le",count=1,fields=fields),separators=(",",":")).encode()
    if len(h)>size:raise ValueError("header too long")
    return topic.encode()+h.ljust(size,b"\0")+b"".join(parts)

# SONIC pose joint_pos is IsaacLab order; feedback/planner qpos is motor/MuJoCo order.
# These names make the conversion auditable instead of applying it twice.
ISAAC_NAMES=[
 "left_hip_pitch_joint","right_hip_pitch_joint","waist_yaw_joint",
 "left_hip_roll_joint","right_hip_roll_joint","waist_roll_joint",
 "left_hip_yaw_joint","right_hip_yaw_joint","waist_pitch_joint",
 "left_knee_joint","right_knee_joint","left_shoulder_pitch_joint","right_shoulder_pitch_joint",
 "left_ankle_pitch_joint","right_ankle_pitch_joint","left_shoulder_roll_joint","right_shoulder_roll_joint",
 "left_ankle_roll_joint","right_ankle_roll_joint","left_shoulder_yaw_joint","right_shoulder_yaw_joint",
 "left_elbow_joint","right_elbow_joint","left_wrist_roll_joint","right_wrist_roll_joint",
 "left_wrist_pitch_joint","right_wrist_pitch_joint","left_wrist_yaw_joint","right_wrist_yaw_joint"]
