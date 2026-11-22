"""Bounded probe scheduling; iteration-derived cursor is reproducible on resume."""
import torch

def probe_tiles(starts, ends, tile):
    a, b = starts.cpu().tolist(), ends.cpu().tolist()
    parts = [torch.arange(s, e - 1, tile) for s, e in zip(a, b)]
    return torch.cat(parts).to(starts.device)

def select_probe_batch(tiles, capacity, sweep):
    if capacity <= 0 or capacity >= len(tiles):
        return tiles
    indices = (torch.arange(capacity, device=tiles.device) + sweep * capacity) % len(tiles)
    return tiles[indices]


def diagnostic_frames(num_frames, count, device):
    """Integer-only inclusive grid: float32 linspace can exceed 2**24-frame libraries."""
    n=min(num_frames,count)
    if n<=0:
        raise ValueError("positive frame and sample counts required")
    return torch.arange(n,device=device,dtype=torch.int64)*(num_frames-1)//max(n-1,1)
