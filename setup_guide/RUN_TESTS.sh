#!/bin/bash
#
# MCTS-DiffMS 测试运行脚本
# 
# 使用方法:
#   bash RUN_TESTS.sh quick      # 快速验证 (30秒)
#   bash RUN_TESTS.sh small      # 小规模测试 5样本 (10分钟)
#   bash RUN_TESTS.sh medium     # 中规模测试 20样本 (40分钟)
#   bash RUN_TESTS.sh full       # 完整测试 100样本 (3-4小时)
#

set -e  # Exit on error

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                    MCTS-DiffMS 测试脚本${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# 检查参数
MODE=${1:-quick}

# 激活环境
echo -e "${YELLOW}激活 conda 环境...${NC}"
source /root/miniforge3/etc/profile.d/conda.sh
conda activate unified-ms-env

# 切换到DiffMS目录
cd /local3/ericjiang/wgc/huaxu/ms/DiffMS

# 根据模式运行测试
case $MODE in
    quick)
        echo -e "${GREEN}运行快速验证测试 (30秒)${NC}"
        echo ""
        python quick_mcts_test.py
        ;;
    
    small)
        echo -e "${GREEN}运行小规模测试 (5样本, 基线 vs MCTS)${NC}"
        echo ""
        
        echo -e "${YELLOW}>>> 基线模式 (不使用MCTS)${NC}"
        python test_mcts_integration.py --num_samples 5 --output_dir results_baseline_small
        
        echo ""
        echo -e "${YELLOW}>>> MCTS模式${NC}"
        python test_mcts_integration.py --num_samples 5 --use_mcts --output_dir results_mcts_small
        
        echo ""
        echo -e "${GREEN}小规模测试完成！${NC}"
        echo -e "基线结果: results_baseline_small/"
        echo -e "MCTS结果: results_mcts_small/"
        ;;
    
    medium)
        echo -e "${GREEN}运行中规模测试 (20样本, 基线 vs MCTS)${NC}"
        echo ""
        
        echo -e "${YELLOW}>>> 基线模式${NC}"
        python test_mcts_integration.py --num_samples 20 --output_dir results_baseline_medium
        
        echo ""
        echo -e "${YELLOW}>>> MCTS模式${NC}"
        python test_mcts_integration.py --num_samples 20 --use_mcts --output_dir results_mcts_medium
        
        echo ""
        echo -e "${GREEN}中规模测试完成！${NC}"
        echo -e "基线结果: results_baseline_medium/"
        echo -e "MCTS结果: results_mcts_medium/"
        ;;
    
    full)
        echo -e "${GREEN}运行完整测试 (100样本, 基线 vs MCTS)${NC}"
        echo -e "${RED}警告: 这将需要 3-4 小时!${NC}"
        echo ""
        read -p "确认继续? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "取消"
            exit 1
        fi
        
        echo -e "${YELLOW}>>> 基线模式${NC}"
        python test_mcts_integration.py --num_samples 100 --output_dir results_baseline_full
        
        echo ""
        echo -e "${YELLOW}>>> MCTS模式${NC}"
        python test_mcts_integration.py --num_samples 100 --use_mcts --output_dir results_mcts_full
        
        echo ""
        echo -e "${GREEN}完整测试完成！${NC}"
        echo -e "基线结果: results_baseline_full/"
        echo -e "MCTS结果: results_mcts_full/"
        ;;
    
    *)
        echo -e "${RED}未知模式: $MODE${NC}"
        echo ""
        echo "用法:"
        echo "  bash RUN_TESTS.sh quick      # 快速验证 (30秒)"
        echo "  bash RUN_TESTS.sh small      # 小规模测试 5样本 (10分钟)"
        echo "  bash RUN_TESTS.sh medium     # 中规模测试 20样本 (40分钟)"
        echo "  bash RUN_TESTS.sh full       # 完整测试 100样本 (3-4小时)"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${GREEN}✓ 完成!${NC}"
echo -e "${BLUE}================================================================================${NC}"

