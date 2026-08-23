#!/bin/bash
# 监控nnU-Net训练进程的脚本

echo "监控TDD牙齿分割模型训练..."
echo "训练已在后台启动，PID: $1"
echo "模型结果将保存在: $DENTALCLAW_HOME/artifacts/models/nnUNet/nnUNet_results/Dataset501_TDDTeethBinary2D"
echo ""

# 创建一个简单的监控脚本
cat << 'EOF' > /tmp/monitor_training.sh
#!/bin/bash
WORK_DIR="$DENTALCLAW_HOME/artifacts/models/nnUNet/nnUNet_results/Dataset501_TDDTeethBinary2D/2d"
LOG_FILE="$WORK_DIR/training_log.txt"

while true; do
    if [ -d "$WORK_DIR" ]; then
        echo "$(date): 检查训练状态..."
        
        # 检查是否有训练日志
        if [ -f "$WORK_DIR/dataset.json" ]; then
            echo "  - 数据集配置文件已生成"
        fi
        
        # 检查checkpoint文件
        if [ -d "$WORK_DIR/checkpoints" ]; then
            echo "  - 检查点目录已创建"
            CHECKPOINTS=$(ls -la $WORK_DIR/checkpoints/ 2>/dev/null | wc -l)
            echo "  - 检查点数量: $((CHECKPOINTS-1))"
        fi
        
        # 检查训练日志
        TRAIN_LOG="$WORK_DIR/trainer_log.json"
        if [ -f "$TRAIN_LOG" ]; then
            EPOCHS=$(jq '.epoch' "$TRAIN_LOG" 2>/dev/null | tail -1)
            if [ ! -z "$EPOCHS" ]; then
                echo "  - 当前训练轮次: $EPOCHS"
            fi
        fi
        
        # 检查是否完成
        if [ -f "$WORK_DIR/checkpoints/best.pth" ] || [ -f "$WORK_DIR/checkpoints/final.pth" ]; then
            echo "  - 模型训练似乎已完成!"
            break
        fi
    else
        echo "  - 训练目录尚未创建，等待中..."
    fi
    
    sleep 60  # 每分钟检查一次
done

echo "训练监控完成"
EOF

chmod +x /tmp/monitor_training.sh
/tmp/monitor_training.sh &
echo "监控后台进程启动"