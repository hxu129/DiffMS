#!/usr/bin/env python3
"""
DiffMS Cache 云端同步工具

支持多种云存储服务：
- AWS S3
- Google Cloud Storage
- 阿里云 OSS
- 腾讯云 COS
- rclone (支持所有 rclone 支持的服务)

使用方法:
    # 上传缓存到云端
    python sync_cache_to_cloud.py upload --provider s3 --bucket my-bucket --remote-path diffms-cache/

    # 从云端下载缓存
    python sync_cache_to_cloud.py download --provider s3 --bucket my-bucket --remote-path diffms-cache/

    # 使用 rclone
    python sync_cache_to_cloud.py upload --provider rclone --remote-name my-remote --remote-path diffms-cache/
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

# 默认缓存目录
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "cache"


def _build_rclone_path(remote_name: str, remote_path: str, bucket: Optional[str] = None) -> str:
    """构建 rclone 路径，自动处理需要 bucket 的服务（如 B2）"""
    # 如果 remote_path 已经包含 bucket（以 bucket_name/ 开头），直接使用
    if remote_path and '/' in remote_path and not remote_path.startswith('/'):
        # 检查是否是 bucket/path 格式
        parts = remote_path.split('/', 1)
        if len(parts) == 2:
            # 可能是 bucket/path 格式，先尝试直接使用
            return f"{remote_name}:{remote_path}"
    
    # 如果指定了 bucket，添加到路径前
    if bucket:
        # 确保路径格式正确
        if remote_path.startswith('/'):
            remote_path = remote_path[1:]
        return f"{remote_name}:{bucket}/{remote_path}"
    
    # 尝试自动检测 bucket（对于 B2）
    try:
        result = subprocess.run(
            ["rclone", "config", "show", remote_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            config = result.stdout
            # 检查是否是 B2
            if "type = b2" in config:
                # 对于 B2，尝试列出 bucket
                list_result = subprocess.run(
                    ["rclone", "lsd", f"{remote_name}:"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if list_result.returncode == 0 and list_result.stdout.strip():
                    # 提取第一个 bucket 名称（通常是默认的）
                    lines = list_result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            # rclone lsd 输出格式: "size date time bucket_name"
                            parts = line.split()
                            if len(parts) >= 4:
                                detected_bucket = parts[-1]
                                if remote_path.startswith('/'):
                                    remote_path = remote_path[1:]
                                return f"{remote_name}:{detected_bucket}/{remote_path}"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # 默认格式
    return f"{remote_name}:{remote_path}"


def check_dependencies(provider: str):
    """检查所需的依赖是否已安装"""
    if provider == "rclone":
        result = subprocess.run(["which", "rclone"], capture_output=True)
        if result.returncode != 0:
            print("错误: 未找到 rclone。请先安装 rclone:")
            print("  curl https://rclone.org/install.sh | sudo bash")
            sys.exit(1)
    elif provider == "s3":
        try:
            import boto3
        except ImportError:
            print("错误: 需要安装 boto3。运行: pip install boto3")
            sys.exit(1)
    elif provider == "gcs":
        try:
            from google.cloud import storage
        except ImportError:
            print("错误: 需要安装 google-cloud-storage。运行: pip install google-cloud-storage")
            sys.exit(1)
    elif provider == "oss":
        try:
            import oss2
        except ImportError:
            print("错误: 需要安装 oss2。运行: pip install oss2")
            sys.exit(1)
    elif provider == "cos":
        try:
            from qcloud_cos import CosConfig, CosS3Client
        except ImportError:
            print("错误: 需要安装 cos-python-sdk-v5。运行: pip install cos-python-sdk-v5")
            sys.exit(1)


def upload_with_rclone(local_path: Path, remote_name: str, remote_path: str, bucket: Optional[str] = None):
    """使用 rclone 上传"""
    # 检查 remote 类型，如果是 B2 或其他需要 bucket 的服务，自动处理
    remote_full_path = _build_rclone_path(remote_name, remote_path, bucket)
    print(f"使用 rclone 上传 {local_path} 到 {remote_full_path}...")
    
    cmd = [
        "rclone", "sync",
        str(local_path),
        remote_full_path,
        "--progress",
        "--transfers", "4",
        "--checkers", "8"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: rclone 上传失败")
        print(f"错误信息: {result.stderr}")
        # 检查是否是 bucket 相关的错误
        if "bucket" in result.stderr.lower() or "must use bucket" in result.stderr.lower():
            print("\n提示: 看起来需要指定 bucket 名称。")
            print("对于 Backblaze B2，路径格式应该是: remote_name:bucket_name/path")
            print("请使用 --bucket 参数指定 bucket 名称，或直接在 --remote-path 中包含 bucket")
            print("例如: --remote-path MS-MCTS/cache 或 --bucket MS-MCTS --remote-path cache")
        sys.exit(1)
    print("上传完成!")


def download_with_rclone(remote_name: str, remote_path: str, local_path: Path, bucket: Optional[str] = None):
    """使用 rclone 下载"""
    # 检查 remote 类型，如果是 B2 或其他需要 bucket 的服务，自动处理
    remote_full_path = _build_rclone_path(remote_name, remote_path, bucket)
    print(f"使用 rclone 从 {remote_full_path} 下载到 {local_path}...")
    
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "rclone", "sync",
        remote_full_path,
        str(local_path),
        "--progress",
        "--transfers", "4",
        "--checkers", "8"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: rclone 下载失败")
        print(f"错误信息: {result.stderr}")
        # 检查是否是 bucket 相关的错误
        if "bucket" in result.stderr.lower() or "must use bucket" in result.stderr.lower():
            print("\n提示: 看起来需要指定 bucket 名称。")
            print("对于 Backblaze B2，路径格式应该是: remote_name:bucket_name/path")
            print("请使用 --bucket 参数指定 bucket 名称，或直接在 --remote-path 中包含 bucket")
            print("例如: --remote-path MS-MCTS/cache 或 --bucket MS-MCTS --remote-path cache")
        sys.exit(1)
    print("下载完成!")


def upload_with_s3(local_path: Path, bucket: str, remote_path: str, 
                   aws_access_key: Optional[str] = None,
                   aws_secret_key: Optional[str] = None,
                   region: str = "us-east-1"):
    """使用 AWS S3 上传"""
    import boto3
    from botocore.exceptions import ClientError
    
    # 创建 S3 客户端
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
    else:
        s3_client = boto3.client('s3', region_name=region)
    
    print(f"上传 {local_path} 到 s3://{bucket}/{remote_path}...")
    
    # 上传所有文件
    uploaded = 0
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = Path(root) / file
            relative_path = local_file.relative_to(local_path)
            s3_key = f"{remote_path.rstrip('/')}/{relative_path}".replace('\\', '/')
            
            try:
                s3_client.upload_file(str(local_file), bucket, s3_key)
                uploaded += 1
                if uploaded % 10 == 0:
                    print(f"已上传 {uploaded} 个文件...")
            except ClientError as e:
                print(f"错误: 上传 {local_file} 失败: {e}")
    
    print(f"上传完成! 共上传 {uploaded} 个文件")


def download_with_s3(bucket: str, remote_path: str, local_path: Path,
                    aws_access_key: Optional[str] = None,
                    aws_secret_key: Optional[str] = None,
                    region: str = "us-east-1"):
    """从 AWS S3 下载"""
    import boto3
    from botocore.exceptions import ClientError
    
    # 创建 S3 客户端
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
    else:
        s3_client = boto3.client('s3', region_name=region)
    
    print(f"从 s3://{bucket}/{remote_path} 下载到 {local_path}...")
    
    # 列出所有文件
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=remote_path.rstrip('/') + '/')
    
    downloaded = 0
    for page in pages:
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            s3_key = obj['Key']
            relative_path = Path(s3_key[len(remote_path.rstrip('/') + '/'):])
            local_file = local_path / relative_path
            
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                s3_client.download_file(bucket, s3_key, str(local_file))
                downloaded += 1
                if downloaded % 10 == 0:
                    print(f"已下载 {downloaded} 个文件...")
            except ClientError as e:
                print(f"错误: 下载 {s3_key} 失败: {e}")
    
    print(f"下载完成! 共下载 {downloaded} 个文件")


def upload_with_gcs(local_path: Path, bucket: str, remote_path: str):
    """使用 Google Cloud Storage 上传"""
    from google.cloud import storage
    
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    
    print(f"上传 {local_path} 到 gs://{bucket}/{remote_path}...")
    
    uploaded = 0
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = Path(root) / file
            relative_path = local_file.relative_to(local_path)
            blob_name = f"{remote_path.rstrip('/')}/{relative_path}".replace('\\', '/')
            
            blob = bucket_obj.blob(blob_name)
            blob.upload_from_filename(str(local_file))
            uploaded += 1
            if uploaded % 10 == 0:
                print(f"已上传 {uploaded} 个文件...")
    
    print(f"上传完成! 共上传 {uploaded} 个文件")


def download_with_gcs(bucket: str, remote_path: str, local_path: Path):
    """从 Google Cloud Storage 下载"""
    from google.cloud import storage
    
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    
    print(f"从 gs://{bucket}/{remote_path} 下载到 {local_path}...")
    
    blobs = bucket_obj.list_blobs(prefix=remote_path.rstrip('/') + '/')
    downloaded = 0
    for blob in blobs:
        relative_path = Path(blob.name[len(remote_path.rstrip('/') + '/'):])
        local_file = local_path / relative_path
        
        local_file.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_file))
        downloaded += 1
        if downloaded % 10 == 0:
            print(f"已下载 {downloaded} 个文件...")
    
    print(f"下载完成! 共下载 {downloaded} 个文件")


def upload_with_oss(local_path: Path, bucket: str, remote_path: str,
                   access_key_id: str, access_key_secret: str, endpoint: str):
    """使用阿里云 OSS 上传"""
    import oss2
    
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket_obj = oss2.Bucket(auth, endpoint, bucket)
    
    print(f"上传 {local_path} 到 oss://{bucket}/{remote_path}...")
    
    uploaded = 0
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = Path(root) / file
            relative_path = local_file.relative_to(local_path)
            object_name = f"{remote_path.rstrip('/')}/{relative_path}".replace('\\', '/')
            
            bucket_obj.put_object_from_file(object_name, str(local_file))
            uploaded += 1
            if uploaded % 10 == 0:
                print(f"已上传 {uploaded} 个文件...")
    
    print(f"上传完成! 共上传 {uploaded} 个文件")


def download_with_oss(bucket: str, remote_path: str, local_path: Path,
                     access_key_id: str, access_key_secret: str, endpoint: str):
    """从阿里云 OSS 下载"""
    import oss2
    
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket_obj = oss2.Bucket(auth, endpoint, bucket)
    
    print(f"从 oss://{bucket}/{remote_path} 下载到 {local_path}...")
    
    downloaded = 0
    for obj in oss2.ObjectIterator(bucket_obj, prefix=remote_path.rstrip('/') + '/'):
        relative_path = Path(obj.key[len(remote_path.rstrip('/') + '/'):])
        local_file = local_path / relative_path
        
        local_file.parent.mkdir(parents=True, exist_ok=True)
        bucket_obj.get_object_to_file(obj.key, str(local_file))
        downloaded += 1
        if downloaded % 10 == 0:
            print(f"已下载 {downloaded} 个文件...")
    
    print(f"下载完成! 共下载 {downloaded} 个文件")


def upload_with_cos(local_path: Path, bucket: str, remote_path: str,
                   secret_id: str, secret_key: str, region: str):
    """使用腾讯云 COS 上传"""
    from qcloud_cos import CosConfig, CosS3Client
    
    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
    client = CosS3Client(config)
    
    print(f"上传 {local_path} 到 cos://{bucket}/{remote_path}...")
    
    uploaded = 0
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = Path(root) / file
            relative_path = local_file.relative_to(local_path)
            key = f"{remote_path.rstrip('/')}/{relative_path}".replace('\\', '/')
            
            client.upload_file(
                Bucket=bucket,
                LocalFilePath=str(local_file),
                Key=key
            )
            uploaded += 1
            if uploaded % 10 == 0:
                print(f"已上传 {uploaded} 个文件...")
    
    print(f"上传完成! 共上传 {uploaded} 个文件")


def download_with_cos(bucket: str, remote_path: str, local_path: Path,
                     secret_id: str, secret_key: str, region: str):
    """从腾讯云 COS 下载"""
    from qcloud_cos import CosConfig, CosS3Client
    
    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
    client = CosS3Client(config)
    
    print(f"从 cos://{bucket}/{remote_path} 下载到 {local_path}...")
    
    # 列出所有文件
    response = client.list_objects(Bucket=bucket, Prefix=remote_path.rstrip('/') + '/')
    downloaded = 0
    
    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            relative_path = Path(key[len(remote_path.rstrip('/') + '/'):])
            local_file = local_path / relative_path
            
            local_file.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(
                Bucket=bucket,
                Key=key,
                DestFilePath=str(local_file)
            )
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"已下载 {downloaded} 个文件...")
    
    print(f"下载完成! 共下载 {downloaded} 个文件")


def main():
    parser = argparse.ArgumentParser(description="DiffMS Cache 云端同步工具")
    subparsers = parser.add_subparsers(dest='action', help='操作')
    
    # 上传命令
    upload_parser = subparsers.add_parser('upload', help='上传缓存到云端')
    upload_parser.add_argument('--provider', required=True, 
                              choices=['s3', 'gcs', 'oss', 'cos', 'rclone'],
                              help='云存储服务提供商')
    upload_parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR,
                              help=f'本地缓存目录 (默认: {DEFAULT_CACHE_DIR})')
    upload_parser.add_argument('--remote-path', required=True,
                              help='云端路径 (例如: diffms-cache/)')
    
    # S3 参数
    upload_parser.add_argument('--bucket', help='存储桶名称 (S3/GCS/OSS/COS 需要)')
    upload_parser.add_argument('--aws-access-key', help='AWS Access Key')
    upload_parser.add_argument('--aws-secret-key', help='AWS Secret Key')
    upload_parser.add_argument('--region', default='us-east-1', help='区域 (默认: us-east-1)')
    
    # rclone 参数
    upload_parser.add_argument('--remote-name', help='rclone remote 名称')
    upload_parser.add_argument('--rclone-bucket', help='Bucket 名称 (B2 等需要 bucket 的服务，或直接在 --remote-path 中使用 bucket/path 格式)')
    
    # OSS 参数
    upload_parser.add_argument('--access-key-id', help='OSS Access Key ID')
    upload_parser.add_argument('--access-key-secret', help='OSS Access Key Secret')
    upload_parser.add_argument('--endpoint', help='OSS Endpoint (例如: oss-cn-hangzhou.aliyuncs.com)')
    
    # COS 参数
    upload_parser.add_argument('--secret-id', help='COS Secret ID')
    upload_parser.add_argument('--secret-key', help='COS Secret Key')
    
    # 下载命令
    download_parser = subparsers.add_parser('download', help='从云端下载缓存')
    download_parser.add_argument('--provider', required=True,
                                choices=['s3', 'gcs', 'oss', 'cos', 'rclone'],
                                help='云存储服务提供商')
    download_parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR,
                                help=f'本地缓存目录 (默认: {DEFAULT_CACHE_DIR})')
    download_parser.add_argument('--remote-path', required=True,
                                help='云端路径 (例如: diffms-cache/)')
    
    # S3 参数
    download_parser.add_argument('--bucket', help='存储桶名称 (S3/GCS/OSS/COS 需要)')
    download_parser.add_argument('--aws-access-key', help='AWS Access Key')
    download_parser.add_argument('--aws-secret-key', help='AWS Secret Key')
    download_parser.add_argument('--region', default='us-east-1', help='区域 (默认: us-east-1)')
    
    # rclone 参数
    download_parser.add_argument('--remote-name', help='rclone remote 名称')
    download_parser.add_argument('--rclone-bucket', help='Bucket 名称 (B2 等需要 bucket 的服务，或直接在 --remote-path 中使用 bucket/path 格式)')
    
    # OSS 参数
    download_parser.add_argument('--access-key-id', help='OSS Access Key ID')
    download_parser.add_argument('--access-key-secret', help='OSS Access Key Secret')
    download_parser.add_argument('--endpoint', help='OSS Endpoint')
    
    # COS 参数
    download_parser.add_argument('--secret-id', help='COS Secret ID')
    download_parser.add_argument('--secret-key', help='COS Secret Key')
    
    args = parser.parse_args()
    
    if not args.action:
        parser.print_help()
        sys.exit(1)
    
    # 检查依赖
    check_dependencies(args.provider)
    
    # 验证参数
    if args.provider == 'rclone':
        if not args.remote_name:
            print("错误: rclone 需要 --remote-name 参数")
            sys.exit(1)
    elif args.provider in ['s3', 'gcs', 'oss', 'cos']:
        if not args.bucket:
            print(f"错误: {args.provider} 需要 --bucket 参数")
            sys.exit(1)
    
    # 对于 rclone，bucket 参数是可选的（用于 B2 等服务）
    rclone_bucket = getattr(args, 'rclone_bucket', None) if args.provider == 'rclone' else None
    
    # 执行操作
    if args.action == 'upload':
        if not args.cache_dir.exists():
            print(f"错误: 缓存目录不存在: {args.cache_dir}")
            sys.exit(1)
        
        if args.provider == 'rclone':
            upload_with_rclone(args.cache_dir, args.remote_name, args.remote_path, rclone_bucket)
        elif args.provider == 's3':
            upload_with_s3(args.cache_dir, args.bucket, args.remote_path,
                          args.aws_access_key, args.aws_secret_key, args.region)
        elif args.provider == 'gcs':
            upload_with_gcs(args.cache_dir, args.bucket, args.remote_path)
        elif args.provider == 'oss':
            if not all([args.access_key_id, args.access_key_secret, args.endpoint]):
                print("错误: OSS 需要 --access-key-id, --access-key-secret, --endpoint 参数")
                sys.exit(1)
            upload_with_oss(args.cache_dir, args.bucket, args.remote_path,
                           args.access_key_id, args.access_key_secret, args.endpoint)
        elif args.provider == 'cos':
            if not all([args.secret_id, args.secret_key]):
                print("错误: COS 需要 --secret-id, --secret-key 参数")
                sys.exit(1)
            upload_with_cos(args.cache_dir, args.bucket, args.remote_path,
                           args.secret_id, args.secret_key, args.region)
    
    elif args.action == 'download':
        if args.provider == 'rclone':
            download_with_rclone(args.remote_name, args.remote_path, args.cache_dir, rclone_bucket)
        elif args.provider == 's3':
            download_with_s3(args.bucket, args.remote_path, args.cache_dir,
                            args.aws_access_key, args.aws_secret_key, args.region)
        elif args.provider == 'gcs':
            download_with_gcs(args.bucket, args.remote_path, args.cache_dir)
        elif args.provider == 'oss':
            if not all([args.access_key_id, args.access_key_secret, args.endpoint]):
                print("错误: OSS 需要 --access-key-id, --access-key-secret, --endpoint 参数")
                sys.exit(1)
            download_with_oss(args.bucket, args.remote_path, args.cache_dir,
                             args.access_key_id, args.access_key_secret, args.endpoint)
        elif args.provider == 'cos':
            if not all([args.secret_id, args.secret_key]):
                print("错误: COS 需要 --secret-id, --secret-key 参数")
                sys.exit(1)
            download_with_cos(args.bucket, args.remote_path, args.cache_dir,
                             args.secret_id, args.secret_key, args.region)


if __name__ == '__main__':
    main()

