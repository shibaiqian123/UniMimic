import argparse
from pathlib import Path
import torch
from blinkmimic.omni_export import load_policy,export
p=argparse.ArgumentParser()
p.add_argument("--ckpt",required=True)
p.add_argument("--out",type=Path,required=True)
a=p.parse_args()
if a.out.exists():p.error("use a new export directory")
ck=torch.load(a.ckpt,map_location="cpu",weights_only=False)
print(export(load_policy(ck),a.out))
