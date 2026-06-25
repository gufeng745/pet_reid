"""
训练监控脚本

实时监控训练进度，显示关键指标
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path


def monitor_training(log_dir='logs/dino', checkpoint_dir='checkpoints/dino'):
    """监控训练进度"""
    print("=" * 60)
    print("Pet Re-ID DINOv3 训练监控")
    print("=" * 60)

    # 查找最新的日志文件
    log_files = list(Path(log_dir).glob('*.log'))
    if not log_files:
        print("未找到日志文件")
        return

    latest_log = max(log_files, key=os.path.getctime)
    print(f"日志文件: {latest_log}")
    print(f"开始监控...")
    print("-" * 60)

    last_epoch = 0
    last_loss = None

    while True:
        try:
            # 读取日志文件
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 查找最新的epoch信息
            for line in reversed(lines):
                if 'Epoch' in line and 'Loss' in line and 'Time' in line:
                    # 解析epoch信息
                    parts = line.split('|')
                    epoch_part = parts[0].strip()
                    loss_part = parts[1].strip()
                    lr_part = parts[2].strip()
                    time_part = parts[3].strip()

                    # 提取数字
                    epoch = int(epoch_part.split('/')[0].split()[-1])
                    total_epochs = int(epoch_part.split('/')[1].split()[0])
                    loss = float(loss_part.split(':')[1].strip())
                    lr = float(lr_part.split(':')[1].strip())
                    time_str = time_part.split(':')[1].strip()

                    if epoch > last_epoch:
                        last_epoch = epoch
                        last_loss = loss

                        # 打印进度
                        now = datetime.now().strftime('%H:%M:%S')
                        print(f"[{now}] Epoch {epoch}/{total_epochs}")
                        print(f"  Loss: {loss:.4f}")
                        print(f"  LR: {lr:.6f}")
                        print(f"  Time: {time_str}")

                        # 检查checkpoint
                        best_model = os.path.join(checkpoint_dir, 'best_dino.pth')
                        if os.path.exists(best_model):
                            size_mb = os.path.getsize(best_model) / (1024*1024)
                            print(f"  Checkpoint: {size_mb:.1f}MB")

                        # 检查是否完成
                        if epoch >= total_epochs:
                            print("\n训练完成!")
                            return

                        print("-" * 60)
                    break

            # 等待5秒后再次检查
            time.sleep(5)

        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception as e:
            print(f"监控错误: {e}")
            time.sleep(5)


if __name__ == '__main__':
    monitor_training()
