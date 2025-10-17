# MCTS-DiffMS Integration Setup Guide

## 概述

本指南介绍如何将 Monte Carlo Tree Search (MCTS) 集成到 DiffMS 中，使用 ICEBERG 作为外部验证器来指导分子生成过程。

## 文件结构

```
DiffMS/
├── src/
│   ├── diffms_mcts.py              # 主模型，包含MCTS实现
│   ├── mcts_verifier.py            # ICEBERG验证器接口
│   └── mcts_utils.py               # 元数据提取工具
├── configs/
│   └── mcts/
│       └── mcts_default.yaml       # MCTS配置文件
├── test_mcts_integration.py        # 完整集成测试脚本
├── quick_mcts_test.py              # 快速验证脚本
└── MCTS_SETUP_GUIDE.md            # 本文档
```

## 核心组件

### 1. MCTS 实现 (`src/diffms_mcts.py`)

**关键修改:**
- 在 `__init__` 中调用 `_init_mcts_config()` (第217行)
- 实现了完整的MCTS算法:
  - `_mcts_select`: 选择节点（UCB策略）
  - `_mcts_expand`: 扩展节点（生成候选分子）
  - `_mcts_evaluate`: 评估节点（使用ICEBERG打分）
  - `_mcts_backup`: 反向传播（更新节点统计）
- `mcts_sample_batch`: 主接口，对batch运行MCTS生成

**使用示例:**
```python
# 启用MCTS的生成
pred_mols, pred_smiles = model.sample_batch(
    batch['graph'],
    return_smiles=True
)
```

### 2. ICEBERG 验证器 (`src/mcts_verifier.py`)

**类 `IcebergVerifier`:**
- 加载旧版ICEBERG模型（gen + inten checkpoints）
- 不需要 `collision_eng` 或 `instrument` 参数
- 使用 matchms 的 CosineGreedy 计算谱相似度

**方法:**
```python
def score(self,
          smiles_list: List[str],
          precursor_mz: float,
          adduct: str,
          instrument: Optional[str],
          collision_eng: Optional[float],
          target_spectra: np.ndarray) -> List[float]:
    """
    返回每个SMILES与目标谱的相似度分数 (0-1)
    """
```

**初始化:**
```python
verifier = IcebergVerifier(
    gen_checkpoint='/path/to/generate.ckpt',
    inten_checkpoint='/path/to/score.ckpt',
    device='cuda',
    tolerance_da=0.01
)
```

### 3. 元数据提取 (`src/mcts_utils.py`)

提供从数据集提取元数据和谱数据的工具:

```python
def extract_metadata_from_spectra_objects(spectra_list, mol_list=None):
    """
    从 Spectra 对象提取:
    - precursor_mz: 前体离子质量
    - adduct: 加合物类型 (如 [M+H]+)
    - instrument: 仪器类型
    - collision_eng: 碰撞能量 (如果有)
    - 原始谱峰数组: (N, 2) [m/z, intensity]
    
    返回: (env_metas, spectra_arrays)
    """
```

### 4. MCTS 配置 (`configs/mcts/mcts_default.yaml`)

```yaml
use_mcts: true                # 启用MCTS
num_simulation_steps: 100     # MCTS模拟步数
branch_k: 5                   # 每步扩展的候选数
c_puct: 1.0                   # UCB探索系数
temp: 1.0                     # Softmax温度
top_p: 0.9                    # nucleus sampling参数
return_topk: 5                # 返回Top-K个结果

verifier_type: 'iceberg'      # 验证器类型

iceberg:
  gen_checkpoint: '/root/ms/ms-pred/quickstart/iceberg/models/canopus_iceberg_generate.ckpt'
  inten_checkpoint: '/root/ms/ms-pred/quickstart/iceberg/models/canopus_iceberg_score.ckpt'

similarity:
  tolerance_da: 0.01          # 谱匹配容差

bins_upper_mz: 1500.0         # 谱binning上限
bins_count: 15000             # bin数量
```

## 安装和设置

### 1. 环境要求

DiffMS 已安装为库在 `unified-ms-env`:

```bash
conda activate unified-ms-env
```

必需的包:
- PyTorch
- PyTorch Geometric
- RDKit
- matchms
- ms-pred (用于ICEBERG)

### 2. 验证设置

运行快速验证脚本:

```bash
cd /root/ms/DiffMS
python quick_mcts_test.py
```

这个脚本会测试:
1. ✓ 所有模块可以正确导入
2. ✓ MCTS配置文件可以加载
3. ✓ ICEBERG验证器可以初始化
4. ✓ DiffMS模型可以加载并启用MCTS

**预期输出:**
```
============================================================
MCTS-DiffMS Quick Integration Test
============================================================

============================================================
TEST 1: Module Imports
============================================================
✓ diffms_mcts imported
✓ mcts_verifier imported
✓ mcts_utils imported
✓ OmegaConf imported

============================================================
TEST 2: MCTS Configuration
============================================================
✓ MCTS config loaded
  use_mcts: True
  num_simulation_steps: 100
  branch_k: 5
  ...

============================================================
TEST 3: ICEBERG Verifier Initialization
============================================================
Initializing ICEBERG verifier...
✓ Verifier initialized successfully
Testing scoring with SMILES: CCO
✓ Scoring works! Score: 0.xxxx

============================================================
TEST 4: DiffMS Model Loading with MCTS
============================================================
Creating datamodule...
✓ Model initialized with MCTS config
  MCTS enabled: True
  MCTS steps: 100
  Branch K: 5
✓ Verifier initialized: IcebergVerifier
✓ Checkpoint loaded successfully
✓ Model set to eval mode

============================================================
TEST SUMMARY
============================================================
IMPORTS: ✓ PASS
CONFIG: ✓ PASS
VERIFIER: ✓ PASS
MODEL: ✓ PASS

============================================================
✓ ALL TESTS PASSED!
You can now run: python test_mcts_integration.py
============================================================
```

## 运行测试

### 快速测试 (5-10个样本)

```bash
# 基线模式 (不使用MCTS)
python test_mcts_integration.py --num_samples 10

# MCTS模式
python test_mcts_integration.py --num_samples 10 --use_mcts
```

### 完整测试 (50-100个样本)

```bash
# 基线
python test_mcts_integration.py --num_samples 100 --output_dir results_baseline

# MCTS
python test_mcts_integration.py --num_samples 100 --use_mcts --output_dir results_mcts
```

### 参数说明

- `--num_samples`: 测试样本数 (默认: 10)
- `--use_mcts`: 启用MCTS引导生成 (默认: False, 使用基线)
- `--seed`: 随机种子 (默认: 42)
- `--output_dir`: 结果保存目录 (默认: 'mcts_test_results')

## 输出和结果

### 结果文件

测试会生成两个文件:

1. **Pickle文件** (`results_<mode>_<timestamp>.pkl`):
   - 完整的结果数据，包含所有预测
   - 可用于后续分析

2. **文本摘要** (`summary_<mode>_<timestamp>.txt`):
   - 人类可读的结果摘要
   - 包含统计数据和每个样本的详细结果

### 评估指标

```python
{
    'num_tested': 10,              # 测试样本数
    'num_successful': 10,          # 成功生成的样本数
    'total_predictions': 100,      # 总预测数 (10样本 × 10预测/样本)
    'total_valid': 98,             # 有效分子数
    'validity_rate': 0.98,         # 有效性: 98%
    'top1_accuracy': 0.15,         # Top-1准确率: 15%
    'avg_top1_similarity': 0.42,   # 平均Top-1 Tanimoto相似度
    'avg_max_similarity': 0.58,    # 平均最大相似度
}
```

### 结果解读

**Top-1 准确率**:
- 理想情况: 5-15%
- MCTS应该能提升这个指标

**Tanimoto 相似度**:
- < 0.3: 差
- 0.3-0.5: 中等
- 0.5-0.7: 好
- > 0.7: 很好

**有效性**:
- 应该 > 90%
- 接近 100% 是理想的

## MCTS 算法细节

### 工作流程

```
1. 初始化: 创建根节点 (t=T, 完全噪声状态)

2. 对于每一步模拟 (共 num_simulation_steps 次):
   a) Selection: 从根开始，使用UCB选择最有希望的路径
      - UCB = exploitation + c_puct × exploration
   
   b) Expansion: 在选中的节点扩展 branch_k 个候选
      - 使用DiffMS的条件概率 p(z_{t-1}|z_t, spectrum)
      - 采样 K 个候选下一状态
   
   c) Evaluation: 对每个候选评估质量
      - 如果到达终止状态 (t=0): 解码为SMILES，用ICEBERG打分
      - 否则: 使用rollout或启发式估计
   
   d) Backup: 反向传播分数
      - 更新路径上所有节点的访问次数和累积奖励

3. 返回: 从根节点选择Top-K个最优路径的终止状态
```

### UCB公式

```python
score = Q(node) + c_puct × P(node) × sqrt(N(parent)) / (1 + N(node))

其中:
- Q(node): 平均奖励 (exploitation)
- P(node): 先验概率 (来自DiffMS)
- N(node): 访问次数
- c_puct: 探索系数 (通常1.0-2.0)
```

### 终止条件

一个节点是终止节点当:
1. 时间步 t = 0 (完全去噪)
2. 可以解码为有效的分子图

## 常见问题

### Q1: ICEBERG加载很慢

**A:** ICEBERG模型较大(~80MB)，首次加载需要10-30秒。这是正常的。

### Q2: MCTS比基线慢多少？

**A:** MCTS大约慢 `num_simulation_steps` × `branch_k` 倍。
- 基线: ~3-5秒/样本
- MCTS (100步×5分支): ~3-5分钟/样本

### Q3: 如何调整MCTS参数？

**A:** 编辑 `configs/mcts/mcts_default.yaml`:
- 增加 `num_simulation_steps` → 更好的搜索，但更慢
- 增加 `branch_k` → 更广的探索，但更慢
- 增加 `c_puct` → 更多探索，更少利用
- 减少 `c_puct` → 更多利用，更少探索

### Q4: 遇到 linalg.eigh 错误怎么办？

**A:** 这是数值不稳定问题，约1-2%的分子会触发。测试脚本会自动跳过这些样本。

如果想修复，可以编辑 `src/diffusion/extra_features.py`:

```python
try:
    eigvals, eigvectors = torch.linalg.eigh(L)
except torch._C._LinAlgError:
    # 添加正则化
    L_reg = L + 1e-6 * torch.eye(L.shape[-1], device=L.device)
    eigvals, eigvectors = torch.linalg.eigh(L_reg)
```

### Q5: 为什么Top-1准确率是0%？

**A:** 可能的原因:
1. **正常现象**: 分子生成任务很难，5-15%是正常水平
2. **样本太少**: 测试10个样本统计不够
3. **模型问题**: 检查checkpoint是否正确加载
4. **应该看Top-5/Top-10**: 真实分子可能在前几名

## 下一步

1. ✅ 完成快速验证 (`quick_mcts_test.py`)
2. ✅ 运行小规模测试 (10样本，基线 vs MCTS)
3. 📊 分析结果，调整参数
4. 🚀 运行大规模测试 (100+样本)
5. 📝 撰写论文，报告结果

## 参考

- **DiffMS 论文**: [链接]
- **ICEBERG 论文**: [链接]
- **MCTS 算法**: Browne et al., "A Survey of Monte Carlo Tree Search Methods"

## 联系

如有问题，请查看:
- GitHub Issues
- 项目文档
- 联系开发者

---

**最后更新**: 2025-10-17

