"""TEST ONLY: exercise distributed local phases on one GPU, NOT a dual-GPU benchmark.
Usage: PYTHONPATH=src python tools/test_distributed_local.py <normal train flags>
"""
import os
import runpy
import sys
if int(os.environ.get("WORLD_SIZE","1"))!=1:
    raise RuntimeError("local-phase smoke harness must not run under torchrun")
import blinkmimic.distributed_bootstrap as bootstrap
import blinkmimic.distributed as runtime
bootstrap.pin_rank=lambda:True
runtime.init=lambda:runtime.Distributed("cuda:0",test=True)
print("TEST ONLY: world_size=1 local phase execution; no NCCL or two-GPU validation",flush=True)
sys.argv[0]="blinkmimic.train"
runpy.run_module("blinkmimic.train",run_name="__main__")
