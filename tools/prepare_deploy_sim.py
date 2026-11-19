"""Build a CPU MuJoCo scene for isolated DDS simulation."""
import argparse
from pathlib import Path
import mujoco
from blinkmimic.deploy.core import Bundle
from blinkmimic.deploy.simulation import CpuSim
ap=argparse.ArgumentParser(); ap.add_argument("--bundle",required=True,type=Path); ap.add_argument("--out",required=True,type=Path)
a=ap.parse_args(); sim=CpuSim(Bundle(a.bundle))
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(sim.spec.to_xml())
# MuJoCo saves relative mesh paths; make this test scene explicit for the local machine.
import xml.etree.ElementTree as ET
from blinkmimic.g1.g1_constants import MJCF_PATH
tree=ET.parse(a.out); root=tree.getroot(); comp=root.find("compiler")
comp.set("meshdir",str(Path(MJCF_PATH).parent.resolve()))
tree.write(a.out,encoding="unicode")
print(a.out)
