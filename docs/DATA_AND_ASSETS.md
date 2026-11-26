# 数据、资产与来源

仓库主要发布代码和文档，不附带原始动作、SMPL模型、实验日志或预训练部署包。
代码许可证不自动覆盖数据、模型、视频和衍生资产；重新分发前须核对各来源的现行条款。

## 数据生成顺序

1. LAFAN1：tools/download_data.sh下载G1重定向动作及机器人描述；脚本中的下载来源为
   https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset 。
   上游注明LAFAN1为CC BY-NC-ND 4.0；不要将其素材标为本项目原创。
2. BONES-SEED：来源https://huggingface.co/datasets/bones-studio/seed 。
   保存实际下载revision；见[bones读取器](bones_reader.md)、[源组划分](bones_subset.md)、[全量](bones_full.md)。
3. 配对SMPL：按[bones_smpl_pairs](bones_smpl_pairs.md)生成共享时段与身份对应清单。
   不将缺失项替换进固定评估集合；原始SMPL来源/版本应记录在生成记录中。
4. 三点：按[threepoint_data](threepoint_data.md)生成左右手与虚拟头部。
   来自G1 FK，不是记录的真实VR轨迹；参考commit与XML指纹随缓存保存。
5. omni：tools/prepare_omni.py消费配对清单和三点缓存；见[入门](GETTING_STARTED.md)。
   --resume复用匹配的完整缓存，不代表恢复训练。

不要公开个人VR原始记录作为默认示例。示例数据需另行确认可分发范围；
目前公开复现依赖使用者自行获取外部数据。

## 代码与运行时来源

BlinkMimic基线来自https://github.com/fuyichen0725/blinkmimic 。
本仓库最初采用下载源码后重新初始化的导入方式；本地基线提交ea61249不是上游commit。
上游精确SHA尚未核实，不伪造对应关系。原README与CITATION保存在docs/upstream。

多模态与实时接入参考作者本地GR00T checkout：
2f07b35b3f3b9053b1f4fd2adeaf56b6bce32366。
该SHA来自本地版本，尚未确认可从NVlabs官方remote获取；仅提供官方URL不代表该本地版本可复现。
要发布完整PICO复现流程，还需给出可获取的fork/patch、环境和资产版本。
三点缓存另保留其实际参考commit，不将两者混为一个版本。

SONIC planner ONNX由运行时从外部GR00T目录读取，不包含在本仓库中；
它与本项目训练的动作encoder/decoder是不同权重。
PICO PC Service、人体解算及其模型资产同样需要独立安装与授权。

## 许可记录

保留LICENSE中的原始版权行；新增代码的维护者标识使用GitHub账号shibaiqian123，不冒用上游作者身份。
具体移植范围见根目录THIRD_PARTY_NOTICES.md与各目录LICENSE。
本次只完成来源梳理，不宣称已完成所有数据、权重和衍生媒体的重新分发许可审查。
