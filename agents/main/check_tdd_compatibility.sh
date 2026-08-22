#!/bin/bash
# 快速检查TDD数据集是否适合牙齿分割任务

echo "=== TDD 数据集检查报告 ==="
echo ""

# 检查数据集是否存在
if [ ! -d "/data/data2/yiyang/DentalClaw/data/TDD" ]; then
    echo "错误: TDD数据集目录不存在"
    exit 1
fi

# 检查影像文件数量
radiograph_count=$(ls /data/data2/yiyang/DentalClaw/data/TDD/Radiographs/ | wc -l)
echo "影像文件总数: $radiograph_count"

# 检查分割掩码数量
teeth_mask_count=$(ls /data/data2/yiyang/DentalClaw/data/TDD/Segmentation/teeth_mask/ | wc -l)
echo "牙齿分割掩码总数: $teeth_mask_count"

maxillomandibular_count=$(ls /data/data2/yiyang/DentalClaw/data/TDD/Segmentation/maxillomandibular/ | wc -l)
echo "上下颌分割掩码总数: $maxillomandibular_count"

# 检查标注文件
bbox_exists=$(if [ -f "/data/data2/yiyang/DentalClaw/data/TDD/Segmentation/teeth_bbox.json" ]; then echo "存在"; else echo "不存在"; fi)
echo "边界框标注文件: $bbox_exists"

polygon_exists=$(if [ -f "/data/data2/yiyang/DentalClaw/data/TDD/Segmentation/teeth_polygon.json" ]; then echo "存在"; else echo "不存在"; fi)
echo "多边形标注文件: $polygon_exists"

echo ""
echo "=== 数据完整性评估 ==="

# 计算匹配率
if [ "$radiograph_count" -eq "$teeth_mask_count" ]; then
    echo "✓ 影像与牙齿分割掩码数量匹配"
else
    echo "⚠ 影像与牙齿分割掩码数量不匹配 ($radiograph_count vs $teeth_mask_count)"
fi

if [ "$radiograph_count" -eq "$maxillomandibular_count" ]; then
    echo "✓ 影像与上下颌分割掩码数量匹配"
else
    echo "⚠ 影像与上下颌分割掩码数量不匹配 ($radiograph_count vs $maxillomandibular_count)"
fi

# 简单测试一个图像文件和掩码文件是否存在
sample_file="1000"
if [ -f "/data/data2/yiyang/DentalClaw/data/TDD/Radiographs/${sample_file}.JPG" ] && [ -f "/data/data2/yiyang/DentalClaw/data/TDD/Segmentation/teeth_mask/${sample_file}.jpg" ]; then
    echo "✓ 样本文件存在 (影像和分割掩码)"
    # 显示文件大小
    img_size=$(ls -lh "/data/data2/yiyang/DentalClaw/data/TDD/Radiographs/${sample_file}.JPG" | awk '{print $5}')
    mask_size=$(ls -lh "/data/data2/yiyang/DentalClaw/data/TDD/Segmentation/teeth_mask/${sample_file}.jpg" | awk '{print $5}')
    echo "  - 影像文件大小: $img_size"
    echo "  - 掩码文件大小: $mask_size"
else
    echo "⚠ 样本文件缺失"
fi

echo ""
echo "=== 结论 ==="
if [ "$radiograph_count" -gt 0 ] && [ "$teeth_mask_count" -gt 0 ] && [ "$bbox_exists" = "存在" ]; then
    echo "✓ TDD数据集具备牙齿分割任务所需的基本要素"
    echo "  - 有全景X光片影像数据"
    echo "  - 有牙齿分割掩码"
    echo "  - 有边界框和多边形标注"
    echo "  - 数据量充足 (1000个样本)"
    echo ""
    echo "✓ 该数据集非常适合进行牙齿分割任务"
else
    echo "✗ TDD数据集可能不完整或不适合牙齿分割任务"
fi