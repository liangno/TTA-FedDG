#!/bin/bash
# =======================
# Script: train_target_fedspl.sh
# Description: Run FedSPL target adaptation for given PACS domain
# =======================

# 第一个参数是 test_domain
#domain=$1

# 指定模型路径（假设 train_pacs.py 输出在当前目录）
# fed_MODEL_DIR="/home/lianghy68/data/Fedspl/main_fedspl/checkpoint/${domain}.pt"

# PORT=10001
# MEMO="target"

# echo ">>> Starting FedSPL for domain ${domain}, using model ${fed_MODEL_DIR}"

# python main_fedspl/main.py \
#     seed=2024 \
#     port=${PORT} \
#     memo=${MEMO} \
#     project="pacs" \
#     data.workers=16 \
#     data.dataset="pacs" \
#     data.fed_domains="[acs]" \
#     data.target_domains="[photo]" \
#     model_fed.arch="resnet18" \
#     model_tta.fed_log_dir=${fed_MODEL_DIR} \
#     optim.lr=2e-4

domain=$1
if [ -z "$domain" ]; then
    echo "[ERROR] Missing domain argument (p, a, c, or s)"
    exit 1
fi

# ========== 映射关系 ==========
# p -> photo
# a -> art_painting
# c -> cartoon
# s -> sketch
case $domain in
    p) target_domain="photo" ;;
    a) target_domain="art_painting" ;;
    c) target_domain="cartoon" ;;
    s) target_domain="sketch" ;;
    *)
        echo "[ERROR] Invalid domain: $domain (must be one of p, a, c, s)"
        exit 1
        ;;
esac
# ==============================

# 模型路径
fed_MODEL_DIR="/home/lianghy68/data/Fedspl/main_fedspl/checkpoint/${domain}.pt"

PORT=10001
MEMO="target"

echo ">>> Running FedSPL for domain=${domain} (${target_domain}), using model=${fed_MODEL_DIR}"

for SEED in 2024
do
    python main_fedspl/main.py \
        seed=${SEED} \
        port=${PORT} \
        memo=${MEMO} \
        project="pacs" \
        data.workers=16 \
        data.dataset="pacs" \
        data.fed_domains="[acs]" \
        data.target_domains="[${target_domain}]" \
        model_fed.arch="resnet18" \
        model_tta.fed_log_dir=${fed_MODEL_DIR} \
        optim.lr=2e-4
done
