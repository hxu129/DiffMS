# MCTS-DiffMS 快速启动指南

## 🚀 5分钟快速开始

### 步骤 1: 激活环境

```bash
cd /root/ms/DiffMS
conda activate unified-ms-env
```

### 步骤 2: 验证设置 (30秒)

```bash
python quick_mcts_test.py
```

**预期输出**:
```
============================================================
✓ ALL TESTS PASSED!
You can now run: python test_mcts_integration.py
============================================================
```

### 步骤 3: 运行快速测试 (5-10分钟)

**基线模式** (不使用MCTS, 更快):
```bash
python test_mcts_integration.py --num_samples 5
```

**MCTS模式** (使用MCTS引导):
```bash
python test_mcts_integration.py --num_samples 5 --use_mcts
```

### 步骤 4: 查看结果

结果保存在 `mcts_test_results/` 目录:
```bash
# 查看文本摘要
cat mcts_test_results/summary_*.txt

# 查看最新结果
ls -lt mcts_test_results/
```

---

## 📊 理解结果

### 关键指标

```yaml
num_tested: 5                    # 测试了5个样本
num_successful: 5                # 5个成功
top1_accuracy: 0.20              # Top-1准确率: 20% (1/5正确)
avg_top1_similarity: 0.45        # 平均相似度: 0.45
avg_max_similarity: 0.62         # 最大相似度: 0.62
validity_rate: 1.00              # 100%生成有效分子
```

### 如何评判

| 指标 | 差 | 中等 | 好 | 说明 |
|------|---|------|----|----|
| Top-1准确率 | <5% | 5-15% | >15% | 第一个预测是否正确 |
| 平均相似度 | <0.3 | 0.3-0.5 | >0.5 | Tanimoto相似度 |
| 有效性 | <90% | 90-95% | >95% | 生成有效分子 |

**重要**: MCTS应该比基线有更高的准确率和相似度！

---

## 🎯 下一步

### 选项 A: 快速实验 (推荐)

```bash
# 测试10个样本，比较基线 vs MCTS
python test_mcts_integration.py --num_samples 10 --output_dir results_baseline
python test_mcts_integration.py --num_samples 10 --use_mcts --output_dir results_mcts
```

### 选项 B: 完整评估

```bash
# 测试100个样本 (需要1-2小时)
python test_mcts_integration.py --num_samples 100 --use_mcts --output_dir results_full
```

### 选项 C: 调参实验

编辑 `configs/mcts/mcts_default.yaml`:

```yaml
# 更多模拟步数 = 更好但更慢
num_simulation_steps: 200  # 默认: 100

# 更多候选 = 更广探索但更慢
branch_k: 10               # 默认: 5

# 更高探索系数 = 更多探索
c_puct: 2.0                # 默认: 1.0
```

然后重新运行:
```bash
python test_mcts_integration.py --num_samples 10 --use_mcts
```

---

## 🔧 遇到问题？

### 测试失败

```bash
# 重新运行验证
python quick_mcts_test.py

# 如果仍然失败，检查环境
conda list | grep torch
conda list | grep rdkit
```

### 内存不足

```bash
# 编辑配置，减少batch size
nano configs/mcts/mcts_default.yaml
# 改: verifier_batch_size: 16
```

### 速度太慢

```bash
# 减少MCTS参数
nano configs/mcts/mcts_default.yaml
# 改: num_simulation_steps: 50
# 改: branch_k: 3
```

---

## 📚 详细文档

- **详细设置**: `MCTS_SETUP_GUIDE.md`
- **实现报告**: `IMPLEMENTATION_COMPLETE.md`
- **原始计划**: `/mcts-diffms-poc.plan.md`

---

## 📞 帮助

如果遇到任何问题:

1. 查看 `MCTS_SETUP_GUIDE.md` 的"故障排除"部分
2. 检查测试输出的错误信息
3. 运行 `python quick_mcts_test.py` 诊断

---

**祝实验顺利! 🎉**

