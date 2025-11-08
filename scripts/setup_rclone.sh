#!/bin/bash
# rclone 快速配置辅助脚本

set -e

echo "=========================================="
echo "rclone 配置辅助脚本"
echo "=========================================="
echo ""

# 检查 rclone 是否安装
if ! command -v rclone &> /dev/null; then
    echo "错误: rclone 未安装"
    echo "请运行以下命令安装:"
    echo "  curl https://rclone.org/install.sh | sudo bash"
    exit 1
fi

echo "rclone 版本: $(rclone version | head -1)"
echo ""

# 显示当前配置的 remote
echo "当前已配置的 remote:"
rclone listremotes 2>/dev/null || echo "  (无)"
echo ""

# 选择配置方式
echo "请选择配置方式:"
echo "1) 交互式配置 (推荐，适合所有云服务)"
echo "2) 查看配置指南"
echo "3) 测试现有 remote"
echo "4) 退出"
echo ""
read -p "请输入选项 [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "启动交互式配置..."
        echo "提示: 如果不知道如何配置，请先选择选项 2 查看配置指南"
        echo ""
        rclone config
        ;;
    2)
        echo ""
        echo "配置指南位置:"
        echo "  DiffMS/scripts/RCLONE_SETUP_GUIDE.md"
        echo ""
        echo "常见云服务配置步骤:"
        echo ""
        echo "【阿里云 OSS】"
        echo "  1. 运行: rclone config"
        echo "  2. 选择: n (新建)"
        echo "  3. 输入名称: alioss"
        echo "  4. 选择类型: oss"
        echo "  5. 输入 AccessKey ID 和 Secret"
        echo "  6. 选择区域 (如: oss-cn-hangzhou)"
        echo ""
        echo "【腾讯云 COS】"
        echo "  1. 运行: rclone config"
        echo "  2. 选择: n (新建)"
        echo "  3. 输入名称: tencentcos"
        echo "  4. 选择类型: cos"
        echo "  5. 输入 Secret ID 和 Key"
        echo "  6. 选择区域 (如: ap-beijing)"
        echo ""
        echo "【AWS S3】"
        echo "  1. 运行: rclone config"
        echo "  2. 选择: n (新建)"
        echo "  3. 输入名称: aws"
        echo "  4. 选择类型: s3"
        echo "  5. 选择提供商: 1 (Amazon S3)"
        echo "  6. 输入 Access Key ID 和 Secret"
        echo "  7. 选择区域"
        echo ""
        echo "【Google Drive】"
        echo "  1. 运行: rclone config"
        echo "  2. 选择: n (新建)"
        echo "  3. 输入名称: gdrive"
        echo "  4. 选择类型: drive"
        echo "  5. 按照浏览器提示完成 OAuth 认证"
        echo ""
        echo "详细配置指南请查看: DiffMS/scripts/RCLONE_SETUP_GUIDE.md"
        ;;
    3)
        echo ""
        REMOTES=$(rclone listremotes 2>/dev/null)
        if [ -z "$REMOTES" ]; then
            echo "错误: 尚未配置任何 remote"
            echo "请先运行选项 1 进行配置"
            exit 1
        fi
        
        echo "已配置的 remote:"
        echo "$REMOTES"
        echo ""
        read -p "请输入要测试的 remote 名称 (不含冒号): " remote_name
        
        if [ -z "$remote_name" ]; then
            echo "错误: remote 名称不能为空"
            exit 1
        fi
        
        echo ""
        echo "测试连接 $remote_name: ..."
        echo ""
        
        # 测试列出目录
        if rclone lsd "${remote_name}:" 2>/dev/null; then
            echo ""
            echo "✓ 连接成功!"
            echo ""
            echo "远程目录列表:"
            rclone lsd "${remote_name}:"
        else
            echo ""
            echo "✗ 连接失败，请检查配置"
            echo ""
            echo "查看配置:"
            rclone config show "$remote_name"
        fi
        ;;
    4)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "配置完成后，可以使用以下命令同步缓存:"
echo ""
echo "  # 上传缓存"
echo "  python DiffMS/scripts/sync_cache_to_cloud.py upload \\"
echo "      --provider rclone \\"
echo "      --remote-name YOUR_REMOTE_NAME \\"
echo "      --remote-path diffms-cache/"
echo ""
echo "  # 下载缓存"
echo "  python DiffMS/scripts/sync_cache_to_cloud.py download \\"
echo "      --provider rclone \\"
echo "      --remote-name YOUR_REMOTE_NAME \\"
echo "      --remote-path diffms-cache/"
echo ""

