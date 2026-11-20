# Third-party notices

Portions of `src/blinkmimic/` and `tools/` come from [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)
(Chong Zhang) and [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (Huy-Quang-Dao), used under their MIT
Licenses, reproduced below.

## InstantTrack

```
MIT License

Copyright (c) 2026 Huy-Quang-Dao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## supermimic_vibed

```
MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dependencies and data

The Python dependencies (mjlab, MuJoCo, MuJoCo Warp, NVIDIA Warp, PyTorch, matplotlib, Pillow) are installed from
their own distributions under their own licenses (Apache-2.0, BSD-3-Clause and others) and are not redistributed here.
The LAFAN1 motions (Ubisoft La Forge, CC BY-NC-ND 4.0) and the Unitree G1 description (BSD-3-Clause) are downloaded
by `tools/download_data.sh` and are not redistributed here either; files derived from LAFAN1 (videos, GIFs,
checkpoints) are not covered by this repository's MIT License.

## SuperMimic-Repro-Sim2Real deployment bridge

DDS bridge sources and wire client in scripts/deploy/bridge, plus adapted state parsing/state-machine behavior, derive from NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real (inspected commit 0905de4). Copyright (c) 2026 Yuxi Liu and Hongyu Ding. MIT license reproduced in scripts/deploy/bridge/LICENSE.

The final sim2sim sensor estimator in src/blinkmimic/deploy/imu_model.py is adapted from the same SuperMimic MIT-licensed project. The simulation DDS adapter reuses Unitree's CRC routine; its BSD-3-Clause license is included in scripts/deploy/bridge/LICENSES/unitree_sdk2-BSD-3-Clause.txt.

## Finite scalar quantization
The scalar bounding/rounding formula in src/blinkmimic/latent.py follows lucidrains/vector-quantize-pytorch 1.27.15 (MIT; licenses/vector-quantize-pytorch-MIT.txt). SONIC configuration reference: NVlabs/GR00T-WholeBodyControl gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml (2 tokens, 32 components, 32 levels). No SONIC model weights are copied. Reference package is test-only and not a deployment dependency.
