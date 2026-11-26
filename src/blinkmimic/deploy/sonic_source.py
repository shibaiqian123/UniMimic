"""SONIC-compatible VR ingress and asynchronous locomotion reference producer.
Uses the existing GR00T PICO streamer unchanged. This process never publishes DDS.
"""
import argparse,concurrent.futures,json,signal,time
from pathlib import Path
import numpy as np
import zmq,msgpack
from .core import Bundle
from .sonic_wire import unpack,ISAAC_NAMES
from .sonic_planner import Planner,initial_context,sample_qpos,planner_window,blend_plan

def aslist(x):
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(type(x).__name__)

class Source:
    def __init__(self):
        self.start=0;self.stop=0;self.planner_mode=True;self.command_time=0.
        self.pose={};self.pose_time=0.;self.last_frame=-1;self.packet_ids=[]
        self.plan={};self.plan_time=0.;self.error=""
        self.epoch=0;self.heading_delta=0.;self.pose_interval=0.;self.pose_packets=0
        self.sensor_time=None;self.pose_cursor=-1
    def accept(self,topic,x,now):
        if topic=="command":
            stop=bool(x.get("stop",[0])[0]);start=bool(x.get("start",[0])[0])
            # Stop wins over a malformed simultaneous start.
            if stop:self.stop+=1
            elif start:self.start+=1
            pm=bool(x.get("planner",[1])[0])
            if pm!=self.planner_mode:
                self.epoch+=1;self.pose.clear();self.sensor_time=None;self.packet_ids=[];self.last_frame=-1;self.pose_time=0.;self.plan_time=0.
            if pm!=self.planner_mode:self.pose_cursor=-1
            self.planner_mode=pm;self.command_time=now
            if "delta_heading" in x:self.heading_delta=float(x["delta_heading"][0])
        elif topic=="planner":
            for key,shape in (("mode",(1,)),("movement",(3,)),("facing",(3,))):
                if key not in x or x[key].shape!=shape:raise ValueError("planner field "+key)
            if "vr_position" in x and (x["vr_position"].shape!=(9,) or x.get("vr_orientation",np.zeros(0)).shape!=(12,)):
                raise ValueError("invalid VR 3-point pose")
            self.plan=x;self.plan_time=now
        elif topic=="pose":
            ids=x.get("frame_index",np.zeros(0)).reshape(-1)
            n=len(ids)
            if not n or x.get("smpl_joints",np.zeros(0)).shape!=(n,24,3) or x.get("body_quat_w",np.zeros(0)).shape!=(n,4) or x.get("joint_pos",np.zeros(0)).shape!=(n,29):
                raise ValueError("expected SONIC pose v3: SMPL [N,24,3], quaternion [N,4], joint_pos [N,29]")
            if int(ids[-1])<=self.last_frame:return # no freshness renewal for duplicate/out-of-order frames
            sensor_time=x.get("timestamp_monotonic")
            if sensor_time is not None:
                sensor_time=float(np.asarray(sensor_time).reshape(-1)[0])
                if not np.isfinite(sensor_time):raise ValueError("invalid pose sensor timestamp")
                if self.sensor_time is not None and sensor_time<=self.sensor_time:return
                self.sensor_time=sensor_time
            for i,frame in enumerate(ids):
                self.pose[int(frame)]=(x["smpl_joints"][i],x["body_quat_w"][i],x["joint_pos"][i])
            self.pose_interval=now-self.pose_time if self.pose_time else 0.
            self.pose_packets+=1
            self.heading_delta+=float(np.asarray(x.get("heading_increment",[0.])).reshape(-1)[0])
            self.last_frame=int(ids[-1]);self.pose_time=now;self.packet_ids=[int(k) for k in ids][-10:]
            self.pose={k:v for k,v in self.pose.items() if k>=self.last_frame-64}
    def pose_window(self, now=None):
        # Keep actual consecutive frames. A bounded nine-frame lookahead delay
        # makes the window chronological; these are not past frames labelled as future.
        if not self.packet_ids:return None
        now=self.pose_time if now is None else now
        first=self.last_frame
        while first-1 in self.pose and self.last_frame-first<9:first-=1
        frame=min(self.last_frame,first+max(0,int((now-self.pose_time)*50+1e-6)))
        frame=max(frame,self.pose_cursor);self.pose_cursor=frame
        ids=list(range(frame,self.last_frame+1))[:10]
        real=len(ids);ids += [ids[-1]]*(10-real)
        values=[self.pose[k] for k in ids]
        return dict(smpl=np.stack([v[0] for v in values]),quat=np.stack([v[1] for v in values]),
                    joints=np.stack([v[2] for v in values]),frame=int(ids[0]),
                    padded_frames=10-real,lookahead_delay_s=(self.last_frame-frame)/50)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stationary-lower-body",action="store_true")
    p.add_argument("--bundle",type=Path,required=True);p.add_argument("--planner-model",type=Path,required=True)
    p.add_argument("--vr-endpoint",default="tcp://127.0.0.1:5556")
    p.add_argument("--feedback-endpoint",default="tcp://127.0.0.1:5557")
    p.add_argument("--reference-bind",default="tcp://127.0.0.1:5558")
    p.add_argument("--record",type=Path,required=True)
    a=p.parse_args()
    b=Bundle(a.bundle);source=Source();planner=Planner(a.planner_model)
    ctx=zmq.Context()
    vr=ctx.socket(zmq.SUB);vr.setsockopt(zmq.RCVHWM,256);vr.setsockopt(zmq.LINGER,0)
    for topic in ("command","planner","pose"):vr.setsockopt_string(zmq.SUBSCRIBE,topic)
    vr.connect(a.vr_endpoint)
    feedback=ctx.socket(zmq.SUB);feedback.setsockopt_string(zmq.SUBSCRIBE,"g1_debug")
    feedback.setsockopt(zmq.CONFLATE,1);feedback.connect(a.feedback_endpoint)
    pub=ctx.socket(zmq.PUB);pub.setsockopt(zmq.SNDHWM,2);pub.setsockopt(zmq.LINGER,0);pub.bind(a.reference_bind)
    a.record.parent.mkdir(parents=True,exist_ok=True)
    fp=a.record.open("wb")
    # Planner reference starts in zero-yaw coordinates, exactly as upstream.
    seq,ms=planner.generate(initial_context(b.q0),{})
    seq_start=time.monotonic();last_plan=seq_start
    executor=concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending=None;pending_start=0.
    stopping=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    if a.stationary_lower_body:print("STATIONARY LOWER BODY: joystick movement/turn commands are ignored; POSE remains full body.",flush=True)
    print("SONIC VR source ready: "+a.vr_endpoint,flush=True)
    try:
        while not stopping[0]:
            now=time.monotonic()
            for _ in range(256):
                try:packet=vr.recv(zmq.NOBLOCK)
                except zmq.Again:break
                fp.write(msgpack.packb(dict(t=now,packet=packet),use_bin_type=True))
                try:
                    topic,values=unpack(packet);source.accept(topic,values,now)
                except Exception as e:source.error=str(e);source.stop+=1
            # Drain feedback for logging/calibration observability; no simulator truth is available here.
            try:
                packet=feedback.recv(zmq.NOBLOCK)
                fp.write(msgpack.packb(dict(t=now,feedback=packet),use_bin_type=True))
            except zmq.Again:pass
            if pending is not None and pending.done():
                try:
                    new,ms=pending.result()
                    seq=blend_plan(seq,seq_start,new,pending_start);seq_start=pending_start
                except Exception as e:source.error="planner: "+str(e);source.stop+=1
                pending=None
            if pending is None and now-last_plan>=.1:
                pending_start=now+.04 # upstream two-control-tick planning lookahead
                context=sample_qpos(seq,((pending_start-seq_start)+np.arange(4)/30)*50)
                # Match upstream idle fallback on a stale planner command.
                command=source.plan if now-source.plan_time<=1. else {}
                if a.stationary_lower_body:command={}
                pending=executor.submit(planner.generate,context,command);last_plan=now
            qpos,qd=planner_window(seq,seq_start,now)
            window=source.pose_window(now)
            mode=("VR_3PT" if "vr_position" in source.plan else
                  "PLANNER_FROZEN_UPPER" if "upper_body_position" in source.plan else "PLANNER") if source.planner_mode else "POSE"
            requested_mode=mode
            if mode=="POSE" and window is None:mode="PLANNER" # hold idle while collecting the causal window
            payload=dict(switch_wait_s=now-source.command_time if mode!=requested_mode else 0.,requested_mode=requested_mode,t=now,start=source.start,stop=source.stop,mode=mode,epoch=source.epoch,error=source.error,
                source_time=max(source.command_time,source.plan_time if source.planner_mode else source.pose_time),
                plan_time=seq_start,plan_end=seq_start+(len(seq)-1)/50,planner_ms=ms,
                qpos=qpos,qd=qd,pose=window,planner=source.plan,
                isaac_joint_names=ISAAC_NAMES,heading_delta=source.heading_delta,
                stationary_lower_body=a.stationary_lower_body,pose_packets=source.pose_packets,
                pose_received_time=source.pose_time,pose_sensor_time=source.sensor_time,pose_interval_s=source.pose_interval,
                pose_padded_frames=window["padded_frames"] if window else 0,pose_delay_s=window["lookahead_delay_s"] if window else 0.,pose_window_policy="buffered_50hz_lookahead_endpoint_hold")
            pub.send(b"reference"+json.dumps(payload,default=aslist,separators=(",",":")).encode(),zmq.NOBLOCK)
            time.sleep(max(0.,.02-(time.monotonic()-now)))
    finally:
        executor.shutdown(wait=True,cancel_futures=True);fp.close()
        for socket in (vr,feedback,pub):socket.close(0)
        ctx.term()
if __name__=="__main__":main()
