#!/usr/bin/env python3
"""
检查TDD数据集的基本信息，验证其是否适合牙齿分割任务
"""

import json
import os
from pathlib import Path

def check_tdd_dataset():
    """检查TDD数据集的基本信息"""
    
    dataset_root = Path("/data/data2/yiyang/DentalClaw/data/TDD")
    
    print("=== TDD 数据集检查报告 ===\n")
    
    # 检查基本目录结构
    radiographs_dir = dataset_root / "Radiographs"
    segmentation_dir = dataset_root / "Segmentation"
    
    if not radiographs_dir.exists():
        print(f"错误: 影像目录不存在 - {radiographs_dir}")
        return
    
    if not segmentation_dir.exists():
        print(f"错误: 分割目录不存在 - {segmentation_dir}")
        return
        
    # 统计影像文件
    radiograph_files = list(radiographs_dir.glob("*.JPG")) + list(radiographs_dir.glob("*.jpg"))
    print(f"影像文件总数: {len(radiograph_files)}")
    
    # 检查分割子目录
    teeth_mask_dir = segmentation_dir / "teeth_mask"
    maxillomandibular_dir = segmentation_dir / "maxillomandibular"
    
    if teeth_mask_dir.exists():
        teeth_mask_files = list(teeth_mask_dir.glob("*.jpg")) + list(teeth_mask_dir.glob("*.JPG"))
        print(f"牙齿分割掩码文件数: {len(teeth_mask_files)}")
    else:
        print("未找到牙齿分割掩码目录")
        teeth_mask_files = []
        
    if maxillomandibular_dir.exists():
        maxillomandibular_files = list(maxillomandibular_dir.glob("*.jpg")) + list(maxillomandibular_dir.glob("*.JPG"))
        print(f"上下颌分割掩码文件数: {len(maxillomandibular_files)}")
    else:
        print("未找到上下颌分割掩码目录")
        maxillomandibular_files = []
    
    # 检查标注文件
    bbox_file = segmentation_dir / "teeth_bbox.json"
    polygon_file = segmentation_dir / "teeth_polygon.json"
    
    print(f"\n=== 标注文件信息 ===")
    
    if bbox_file.exists():
        print(f"边界框标注文件存在: {bbox_file.name}")
        with open(bbox_file, 'r') as f:
            try:
                bbox_data = json.load(f)
                print(f"边界框标注条目数: {len(bbox_data)}")
                
                # 检查第一个样本的标注信息
                if len(bbox_data) > 0:
                    first_sample = bbox_data[0]
                    objects = first_sample.get('Label', {}).get('objects', [])
                    print(f"首个样本的边界框数量: {len(objects)}")
                    
                    if len(objects) > 0:
                        first_obj = objects[0]
                        print(f"首个边界框示例: 牙齿 {first_obj['title']}: {first_obj['bounding box']}")
                        
            except Exception as e:
                print(f"读取边界框文件时出错: {e}")
    else:
        print("未找到边界框标注文件")
    
    if polygon_file.exists():
        print(f"多边形标注文件存在: {polygon_file.name}")
        with open(polygon_file, 'r') as f:
            try:
                polygon_data = json.load(f)
                print(f"多边形标注条目数: {len(polygon_data)}")
                
                # 检查第一个样本的标注信息
                if len(polygon_data) > 0:
                    first_sample = polygon_data[0]
                    objects = first_sample.get('Label', {}).get('objects', [])
                    print(f"首个样本的多边形对象数量: {len(objects)}")
                    
                    if len(objects) > 0:
                        first_obj = objects[0]
                        has_polygons = 'polygons' in first_obj
                        print(f"首个对象是否包含多边形数据: {has_polygons}")
                        
            except Exception as e:
                print(f"读取多边形文件时出错: {e}")
    else:
        print("未找到多边形标注文件")
    
    # 验证影像与标注的匹配情况
    print(f"\n=== 匹配验证 ===")
    
    radiograph_names = {f.stem for f in radiograph_files}
    teeth_mask_names = {f.stem for f in teeth_mask_files}
    maxillomandibular_names = {f.stem for f in maxillomandibular_files}
    
    print(f"影像文件名集合大小: {len(radiograph_names)}")
    print(f"牙齿掩码文件名集合大小: {len(teeth_mask_names)}")
    print(f"上下颌掩码文件名集合大小: {len(maxillomandibular_names)}")
    
    missing_masks = radiograph_names - teeth_mask_names
    if len(missing_masks) == 0:
        print("✓ 所有影像都有对应的牙齿分割掩码")
    else:
        print(f"⚠ {len(missing_masks)} 个影像缺少牙齿分割掩码")
    
    # 总结
    print(f"\n=== 总结 ===")
    print(f"TDD数据集适合牙齿分割任务吗？")
    
    requirements_met = 0
    total_requirements = 4
    
    if len(radiograph_files) > 0:
        print("- ✓ 存在影像数据")
        requirements_met += 1
    else:
        print("- ✗ 缺少影像数据")
    
    if len(teeth_mask_files) > 0:
        print("- ✓ 存在牙齿分割掩码")
        requirements_met += 1
    else:
        print("- ✗ 缺少牙齿分割掩码")
    
    if len(missing_masks) == 0:
        print("- ✓ 影像与掩码完全匹配")
        requirements_met += 1
    else:
        print("- ⚠ 影像与掩码不完全匹配")
    
    if bbox_file.exists() or polygon_file.exists():
        print("- ✓ 存在精确标注数据")
        requirements_met += 1
    else:
        print("- ⚠ 缺少精确标注数据")
    
    print(f"\n合规度: {requirements_met}/{total_requirements}")
    
    if requirements_met >= 3:
        print("✓ TDD数据集基本适合牙齿分割任务")
    else:
        print("✗ TDD数据集不适合牙齿分割任务")
    
    return {
        "radiograph_count": len(radiograph_files),
        "teeth_mask_count": len(teeth_mask_files),
        "maxillomandibular_count": len(maxillomandibular_files),
        "bbox_annotations": bbox_file.exists(),
        "polygon_annotations": polygon_file.exists(),
        "completeness_ratio": len(teeth_mask_names)/len(radiograph_names) if len(radiograph_names) > 0 else 0
    }

if __name__ == "__main__":
    check_tdd_dataset()