"""属性标注数据集验证脚本

在训练前运行此脚本，检查数据集的完整性：
1. 图片文件是否存在且可读取
2. 标签值是否有效
3. 数据分布是否均衡

用法：
    python validate_attr_data.py
"""

import os
import csv
import json
from collections import Counter
from PIL import Image


def find_image_dir():
    """查找图片目录"""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'cat_dog_attr')
    train_dir = os.path.join(base, 'train')
    if os.path.isdir(train_dir):
        return train_dir
    return base


def load_annotations(csv_path):
    """加载标注文件"""
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def validate_dataset(csv_path, image_dir):
    """完整验证数据集
    
    Returns:
        report: 验证报告字典
    """
    rows = load_annotations(csv_path)
    
    # 收集所有类别值
    color_primary_vals = Counter()
    color_secondary_vals = Counter()
    pattern_vals = Counter()
    
    for row in rows:
        color_primary_vals[row['color_primary'].strip()] += 1
        for item in row['color_secondary'].split(','):
            if item.strip():
                color_secondary_vals[item.strip()] += 1
        for item in row['pattern'].split(','):
            if item.strip():
                pattern_vals[item.strip()] += 1
    
    # 构建类别集合
    all_colors = set(color_primary_vals.keys()) | set(color_secondary_vals.keys())
    all_patterns = set(pattern_vals.keys())
    
    # 验证每行数据
    valid_rows = []
    missing_images = []
    invalid_labels = []
    
    for i, row in enumerate(rows):
        filename = row.get('filename', f'row_{i}')
        img_path = os.path.join(image_dir, filename)
        
        # 检查图片
        if not os.path.exists(img_path):
            missing_images.append(filename)
            continue
        
        try:
            with Image.open(img_path) as img:
                img.load()
        except Exception as e:
            missing_images.append(f"{filename} (损坏：{e})")
            continue
        
        # 检查标签
        row_valid = True
        
        color_pri = row.get('color_primary', '').strip()
        if not color_pri:
            invalid_labels.append((filename, 'color_primary', '空值'))
            row_valid = False
        elif color_pri not in all_colors:
            invalid_labels.append((filename, 'color_primary', f'未知：{color_pri}'))
            row_valid = False
        
        pattern = row.get('pattern', '').strip()
        if not pattern:
            invalid_labels.append((filename, 'pattern', '空值'))
            row_valid = False
        else:
            for item in pattern.split(','):
                item = item.strip()
                if item and item not in all_patterns:
                    invalid_labels.append((filename, 'pattern', f'未知：{item}'))
                    row_valid = False
                    break
        
        if row_valid:
            valid_rows.append(row)
    
    # 计算有效数据的分布
    valid_color_pri = Counter()
    valid_color_sec = Counter()
    valid_pattern = Counter()
    
    for row in valid_rows:
        valid_color_pri[row['color_primary'].strip()] += 1
        for item in row['color_secondary'].split(','):
            if item.strip():
                valid_color_sec[item.strip()] += 1
        for item in row['pattern'].split(','):
            if item.strip():
                valid_pattern[item.strip()] += 1
    
    # 生成报告
    report = {
        'summary': {
            'total_samples': len(rows),
            'valid_samples': len(valid_rows),
            'missing_images': len(missing_images),
            'invalid_labels': len(invalid_labels),
            'valid_percentage': len(valid_rows) / len(rows) * 100 if rows else 0,
        },
        'color_primary_distribution': dict(valid_color_pri),
        'color_secondary_distribution': dict(valid_color_sec),
        'pattern_distribution': dict(valid_pattern),
        'num_color_classes': len(all_colors),
        'num_pattern_classes': len(all_patterns),
        'color_classes': sorted(all_colors),
        'pattern_classes': sorted(all_patterns),
        'missing_image_list': missing_images[:100],
        'invalid_label_list': [list(x) for x in invalid_labels[:100]],
    }
    
    return report


def print_report(report):
    """打印验证报告"""
    print("\n" + "=" * 60)
    print("属性标注数据集验证报告")
    print("=" * 60)
    
    s = report['summary']
    print(f"\n【总体统计】")
    print(f"  总样本数：{s['total_samples']}")
    print(f"  有效样本：{s['valid_samples']} ({s['valid_percentage']:.1f}%)")
    print(f"  缺失/损坏图片：{s['missing_images']}")
    print(f"  无效标签：{s['invalid_labels']}")
    
    print(f"\n【类别统计】")
    print(f"  颜色类别数：{report['num_color_classes']}")
    print(f"  花纹类别数：{report['num_pattern_classes']}")
    
    print(f"\n【主色分布】")
    for color, count in sorted(report['color_primary_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {color}: {count}")
    
    print(f"\n【副色分布】")
    for color, count in sorted(report['color_secondary_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {color}: {count}")
    
    print(f"\n【花纹分布】")
    for pattern, count in sorted(report['pattern_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count}")
    
    if report['missing_image_list']:
        print(f"\n【缺失/无效文件】(前 20 个)")
        for f in report['missing_image_list'][:20]:
            print(f"  - {f}")
        if len(report['missing_image_list']) > 20:
            print(f"  ... 还有 {len(report['missing_image_list']) - 20} 个")
    
    print("\n" + "=" * 60)


def main():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotations.csv')
    image_dir = find_image_dir()
    
    print(f"标注文件：{csv_path}")
    print(f"图片目录：{image_dir}")
    
    if not os.path.exists(csv_path):
        print(f"错误：找不到标注文件 {csv_path}")
        return
    
    print("\n正在验证数据集...")
    report = validate_dataset(csv_path, image_dir)
    
    # 打印报告
    print_report(report)
    
    # 保存报告
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_validation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n验证报告已保存到：{report_path}")
    
    # 退出码
    if report['summary']['valid_samples'] == 0:
        print("\n错误：没有有效样本，无法训练")
        exit(1)
    elif report['summary']['valid_percentage'] < 90:
        print(f"\n警告：有效样本比例较低 ({report['summary']['valid_percentage']:.1f}%)")
        exit(0)
    else:
        print(f"\n数据集验证通过！")
        exit(0)


if __name__ == '__main__':
    main()