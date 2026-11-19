"""BlinkMimic runtime. Rotation matrices map body vectors to world vectors."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

JOINTS = [side + "_" + part + "_joint" for side in ("left", "right")
          for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
JOINTS += ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
JOINTS += [side + "_" + part + "_joint" for side in ("left", "right")
           for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                        "wrist_roll", "wrist_pitch", "wrist_yaw")]
OBS_LAYOUT = "q-q0,dq,gyro_pelvis,gravity_pelvis,last_raw_action,reference_rotation6d_rowmajor"
STAGES = {
    "air": dict(max_tilt=0.35, velocity_fraction=0.3, torque_fraction=0.5, max_action=10.0),
    "feet": dict(max_tilt=0.35, velocity_fraction=0.3, torque_fraction=0.5, max_action=10.0),
    "slack": dict(max_tilt=0.5, velocity_fraction=0.5, torque_fraction=1.0, max_action=15.0),
    "ground": dict(max_tilt=0.8, velocity_fraction=0.8, torque_fraction=1.0, max_action=50.0),
}
LIMITS = dict(max_temperature=75.0, state_age_s=0.05, policy_timeout_s=0.02,
              interpolation_s=4.0, standby_s=3.0, standby_tilt=0.14, standby_dq=1.0)

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 0.05:
        raise ValueError("invalid wxyz quaternion")
    w,x,y,z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def yaw(R):
    return np.arctan2(R[1,0], R[0,0])

def yaw_rotation(a):
    c,s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])

class Bundle:
    def __init__(self, path):
        self.path = Path(path)
        self.cfg = json.loads((self.path / "deploy_config.json").read_text())
        c = self.cfg
        if c["schema"] != "blinkmimic_single_motion_v1" or c["joint_names"] != JOINTS:
            raise ValueError("unsupported schema / motor order")
        if (c["history"], c["goal_frames"], c["obs_dim"], c["goal_dim"], c["latent_dim"]) != (10,10,99,730,32):
            raise ValueError("unsupported model dimensions")
        if c["control_dt"] != 0.02 or c["obs_layout"] != OBS_LAYOUT:
            raise ValueError("unsupported observation / timing contract")
        gain=c.get("preparation_gain_scale",1.0)
        if not np.isfinite(gain) or gain<=0:
            raise ValueError("unsupported preparation gain scale")
        for name in ("policy.onnx", "motion.npz"):
            if digest(self.path/name) != c["sha256"][name]:
                raise ValueError("bundle hash mismatch: " + name)
        for key in ("q0","scale","kp","kd","effort","lower","upper","velocity"):
            a = np.asarray(c[key], float)
            if a.shape != (29,) or not np.isfinite(a).all():
                raise ValueError("invalid " + key)
            setattr(self, key, a)
        if (self.kp <= 0).any() or (self.kd < 0).any() or (self.scale <= 0).any() or (self.lower >= self.upper).any() or (self.effort <= 0).any() or (self.velocity <= 0).any():
            raise ValueError("invalid robot limits / gains")
        with np.load(self.path/"motion.npz", allow_pickle=False) as z:
            self.motion = {k: z[k].copy() for k in z.files}
        self.T = len(self.motion["q"])
        shapes = {"q":(self.T,29),"root_R":(self.T,3,3),"goal":(self.T,730),"static_goal":(2,730)}
        if self.T < 2:
            raise ValueError("motion too short")
        for k, shape in shapes.items():
            if self.motion[k].shape != shape or not np.isfinite(self.motion[k]).all():
                raise ValueError("invalid motion " + k)
        for R in self.motion["root_R"]:
            if not np.allclose(R.T@R,np.eye(3),atol=1e-4) or abs(np.linalg.det(R)-1)>1e-4:
                raise ValueError("invalid reference rotation")
        self.fingerprint = digest(self.path/"deploy_config.json")

class OnnxPolicy:
    def __init__(self, bundle):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(bundle.path/"policy.onnx"), so, providers=["CPUExecutionProvider"])
        ins = {i.name:i.shape for i in self.session.get_inputs()}
        outs={o.name:o.shape for o in self.session.get_outputs()}
        if outs.get("actions",[None,None])[1:] != [29]:
            raise ValueError("invalid ONNX action output")
        if set(ins) != {"obs_history","goal"} or ins["obs_history"][1:] != [10,99] or ins["goal"][1:] != [730]:
            raise ValueError("invalid ONNX inputs")
    def __call__(self, history, goal):
        return self.session.run(["actions"], {"obs_history":history[None].astype(np.float32),
                                             "goal":goal[None].astype(np.float32)})[0][0]

class DeployCore:
    def __init__(self, bundle, policy=None):
        self.b = bundle
        self.policy = policy if policy is not None else OnnxPolicy(bundle)
        self.frame = 0
        self.phase = "pre"
        self.history = None
        self.last_action = np.zeros(29)
        self.heading = None

    def align(self, R_robot):
        if self.heading is not None:
            raise ValueError("heading is fixed for this session")
        self.heading = yaw_rotation(yaw(R_robot)-yaw(self.b.motion["root_R"][0]))

    def start_motion(self):
        if self.phase != "pre":
            raise ValueError("single playback only")
        self.phase, self.frame = "motion", 0

    def reference(self):
        if self.phase == "pre":
            return 0, self.b.motion["static_goal"][0]
        if self.phase == "post":
            return self.b.T-1, self.b.motion["static_goal"][1]
        return self.frame, self.b.motion["goal"][self.frame]

    def observe(self, q, dq, gyro, R):
        if self.heading is None:
            raise ValueError("call align first")
        f, _ = self.reference()
        rel = R.T @ self.heading @ self.b.motion["root_R"][f]
        o = np.concatenate((q-self.b.q0,dq,gyro,-R[2],self.last_action,rel[:,:2].reshape(6)))
        if o.shape != (99,) or not np.isfinite(o).all():
            raise ValueError("invalid observation")
        self.history = np.repeat(o[None],10,axis=0) if self.history is None else np.concatenate((self.history[1:],o[None]))
        return self.history

    def step(self, q, dq, gyro, R):
        f,g = self.reference()
        hist = self.observe(np.asarray(q),np.asarray(dq),np.asarray(gyro),R)
        t = time.perf_counter()
        a = np.asarray(self.policy(hist,g),float)
        ms = (time.perf_counter()-t)*1000
        if a.shape != (29,) or not np.isfinite(a).all():
            raise ValueError("invalid policy action")
        self.last_action = a.copy()
        result = dict(action=a, target=self.b.q0+self.b.scale*a, frame=f, phase=self.phase, inference_ms=ms)
        if self.phase == "motion":
            if self.frame >= self.b.T-2:
                self.frame = self.b.T-1
                self.phase = "post"
            else:
                self.frame += 1
        return result

def sensor_transform(profile):
    if profile.get("imu_frame") != "pelvis_rigid" or profile.get("quaternion_order") != "wxyz" or profile.get("gyro_units") != "rad/s":
        raise ValueError("explicit pelvis-rigid IMU profile required; moving torso IMU unsupported")
    return rotation(profile["R_pelvis_from_imu_wxyz"])

def transform_imu(quat, gyro, R_pelvis_from_imu):
    # R_world_imu = R_world_pelvis @ R_pelvis_imu.
    return rotation(quat) @ R_pelvis_from_imu.T, R_pelvis_from_imu @ np.asarray(gyro,float)

def state_guard(s, now, b, stage, limits=LIMITS, check_velocity=True):
    for k in ("q","dq","gyro","quat","temp"):
        if not np.isfinite(s[k]).all():
            return "nonfinite " + k
    if s["mode_machine"] != 5 or s["mode_pr"] != 0:
        return "requires full 29DoF G1 PR mode"
    if now-s["t_rx"] > limits["state_age_s"] or now-s["t_tick_change"] > limits["state_age_s"]:
        return "stale state"
    if np.any(s["motorstate"]):
        return "motor fault"
    if np.max(s["temp"]) >= limits["max_temperature"]:
        return "overtemperature"
    if check_velocity and (np.abs(s["dq"]) > b.velocity*stage["velocity_fraction"]).any():
        return "joint overspeed"
    try:
        rotation(s["quat"])
    except ValueError:
        return "invalid quaternion"
    return None

def limit_target(target,q,dq,b,stage,kp=None,kd=None):
    kp=b.kp if kp is None else kp
    kd=b.kd if kd is None else kd
    tau = b.effort*stage["torque_fraction"]
    low = q+(-tau+kd*dq)/kp
    high = q+(tau+kd*dq)/kp
    if stage.get("clamp_positions",False):
        low=np.maximum(b.lower,low);high=np.minimum(b.upper,high)
    if (low>high).any():
        raise ValueError("no feasible torque-limited target")
    out = np.clip(target,low,high)
    return out,int(np.count_nonzero(abs(out-target)>1e-9))
