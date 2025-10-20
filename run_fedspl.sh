#!/bin/bash
# =======================
# Script: run_pacs_and_fedspl_test.sh
# Description: Run FedAvg on PACS dataset for all domains (p, a, c, s),
#              then run the target_fedspl script for each domain,
#              and clean up log_dir at the end.
# =======================

# 激活环境（如有需要）
# source activate your_env_name
# 或 module load cuda/conda 等

# 域名列表
domains=("p" "a" "c" "s")

# 循环遍历每个域名
for domain in "${domains[@]}"; do
    echo "==============================================="
    echo ">>> Running PACS training with test_domain=${domain}"
    echo "==============================================="

    # 运行 FedAvg 训练
    python algorithms/fedspl/train_pacs.py \
        --test_domain "$domain" \
        --lr 0.001 \
        --batch_size 256 \
        --comm 200 \
        --model resnet18 \
        --note debug

    # 检查执行结果
    if [ $? -ne 0 ]; then
        echo "[ERROR] train_pacs.py failed for domain ${domain}"
        exit 1
    fi
    echo "[OK] train_pacs.py finished for domain ${domain}"

    # ===============================================
    # 运行 FedSPL 目标自适应
    # ===============================================
    echo ">>> Running train_target_fedspl.sh for domain ${domain}"

    # 把当前域名传给 train_target_fedspl.sh
    bash ./main_fedspl/train_target_fedspl.sh "$domain"

    if [ $? -ne 0 ]; then
        echo "[ERROR] train_target_fedspl.sh failed for domain ${domain}"
        exit 1
    fi
    echo "[OK] train_target_fedspl.sh finished for domain ${domain}"
done

# ===============================================
# 删除 checkpoint 文件夹下的所有文件
# ===============================================
CHECKPOINT_DIR="main_fedspl/checkpoint"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Deleting all contents under $CHECKPOINT_DIR..."
    rm -rf "${CHECKPOINT_DIR:?}/"*
    echo "✅ All files under $CHECKPOINT_DIR deleted."
else
    echo "⚠️ No checkpoint folder found to delete."
fi

