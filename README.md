# PorousRT

PorousRT 是一个面向孔隙尺度反应运移建模的 Python 框架。当前版本将
变密度 Darcy--Brinkman--Stokes 流动、FiPy 守恒组分运移和 PhreeqcRM
局部化学反应通过顺序非迭代耦合（SNIA）组织在一起，并让矿物体积变化
反馈到孔隙率、流体物性和流场。

## 当前内容

```text
porousrt/                         可复用耦合框架
cases/halite_dissolution/         石盐溶解案例
cases/carnallite_replacement/     光卤石—钾石盐—石盐案例
cases/carnallite_replacement_binary/  随机二值光卤石几何案例
cases/carnallite_wormholing/      混合蒸发盐孔隙含水层优势通道与补给水流失案例
cases/carnallite_wormholing_equilibrium/  三种矿物全部采用局部平衡反应的对照案例
cases/carnallite_wormholing_no_primary_sylvite/  无原生钾石盐、双倍光卤石速率变体
cases/two_inlet_barite_precipitation/  双入口重晶石沉淀案例
tests/                            当前框架的单元测试
ref/                              原始论文（受保护，不由程序写入）
outputs/                          历史正式结果（受保护，不由程序写入）
```

框架的主要职责如下：

- `grid.py` 与 `geometry.py`：规则网格和弥散界面方块几何；
- `flow.py` 与 `boundary_flow.py`：默认及通用矩形边界的变密度 Darcy--Brinkman--Stokes 求解；
- `transport.py`：支持独立多入口组分向量的 FiPy 多组分守恒运移；
- `coupling.py`：流动—运移—化学的 SNIA 调度；
- `artifacts.py`：逐时间步完整场与清单的通用持久化；
- `media.py`：MP4/GIF 编码；
- 各案例的 `chemistry.py`：PhreeqcRM 相平衡、矿物体积和流体物性。

更详细的依赖关系见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 环境

项目使用已有的 `work` Conda 环境：

```powershell
conda run -n work python -c "import fipy, phreeqcrm; print('ready')"
```

依赖也记录在 `requirements.txt` 与 `pyproject.toml` 中。

## 运行案例

快速验证：

```powershell
conda run -n work python -m cases.halite_dissolution --quick
conda run -n work python -m cases.carnallite_replacement --quick
conda run -n work python -m cases.carnallite_wormholing --quick
conda run -n work python -m cases.carnallite_wormholing_equilibrium --quick
conda run -n work python -m cases.carnallite_wormholing_no_primary_sylvite --quick
conda run -n work python -m cases.two_inlet_barite_precipitation --quick
```

`--quick` 用于数值烟雾测试，会保留帧图、快照和数据，但跳过 MP4/GIF
编码。完整计算去掉 `--quick`，并默认生成视频和动图。结果分别写入：

```text
cases/halite_dissolution/results/latest/
cases/carnallite_replacement/results/latest/
cases/carnallite_wormholing/results/latest/
cases/carnallite_wormholing_equilibrium/results/latest/
cases/carnallite_wormholing_no_primary_sylvite/results/latest/
cases/two_inlet_barite_precipitation/results/latest/
```

可以使用 `--output` 指定案例目录下的另一个结果子目录。每次结果包含
指标 CSV、最终场、快照、汇总图、MP4/GIF，以及实际提交给 PhreeqcRM 的
持久化 `.pqi` 和初始化 YAML。

## 测试

```powershell
conda run -n work python -m unittest discover -s tests -v
```

测试覆盖网格和几何、默认与多边界变密度流动、多入口 FiPy 运移、SNIA 调度、矿物化学
以及视频编码参数。`ref/` 和 `outputs/` 不参与测试，也不会被测试改写。

## 模型边界

石盐与原光卤石置换案例采用局部平衡化学；基准优势通道案例显式解析连续孔隙、
不可溶骨架、光卤石、原生钾石盐和石盐，三种反应矿物均采用统一可逆动力学，
并用 Br 示踪注入水。`carnallite_wormholing_equilibrium` 保留同一工况，但把
三种矿物全部改为有限库存局部平衡相；双入口重晶石案例同时提供平衡与示范性动力学模式。
动力学默认值并非经过实验标定的表面反应模型；
弥散界面会把亚网格矿物体积映射为连续孔隙率。该实现适合算法研究与案例
验证，若用于定量预测，仍需做网格/时间步收敛、动力学标定和守恒审计。
