"""GPU selection before importing torch, latent, tensorboard or Warp."""
import os

def pin_rank():
    world=int(os.environ.get("WORLD_SIZE","1"))
    if world==1:return False
    if world!=2 or int(os.environ.get("LOCAL_WORLD_SIZE",str(world)))!=world:
        raise ValueError("only single-node two-GPU training is supported")
    rank=int(os.environ["LOCAL_RANK"])
    visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    devices=[v.strip() for v in visible.split(",")] if visible is not None else ["0","1"]
    if len(devices)<world or len(set(devices))!=len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must expose two distinct GPUs")
    os.environ["CUDA_VISIBLE_DEVICES"]=devices[rank]
    return True
