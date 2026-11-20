"""Bounded retries for a seqlock copy collision; never use a failed copy."""
import time

def read_snapshot(bridge, buffer, retry_s=.002, clock=time.monotonic, sleep=time.sleep):
    until=clock()+retry_s
    while True:
        if bridge.read_snap(buffer):
            return bool(buffer[0]["n_rx"])
        if clock()>=until:
            return False
        sleep(.0001)  # let the preempted writer finish; avoid a busy-spin
