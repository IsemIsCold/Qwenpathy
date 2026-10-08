#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/train_venv/bin:$PATH
cd /mnt/storage
grep -q '"migu_train"' sft/dataset_info.json || { echo "migu_train missing in dataset_info.json"; exit 1; }
echo "train start $(date)"
FORCE_TORCHRUN=1 llamafactory-cli train configs/full.yaml > runs/full.log 2>&1
echo "merge start $(date)"
llamafactory-cli export configs/merge_full.yaml > runs/merge_full.log 2>&1
cd /root/merged/full
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json added_tokens.json special_tokens_map.json; do
  if [ -f /root/participant/model/$f ]; then cp /root/participant/model/$f .; fi
done
rm -f chat_template.jinja
for s in n50 n50b; do
  echo "eval $s $(date)"
  VLLM_MODEL_DIR=/root/merged/full VLLM_SERVED_MODEL_NAME=model bash /root/participant/start.sh /mnt/storage/val/val_input_$s.jsonl /mnt/storage/out/full_$s > /mnt/storage/out/full_$s.run.log 2>&1
  /opt/train_venv/bin/python /mnt/storage/scripts/eval.py /mnt/storage/out/full_$s/submission.jsonl /mnt/storage/val/val_gold_$s.jsonl --out /mnt/storage/out/full_$s/metrics.json --history "" --force > /mnt/storage/out/full_$s.eval.txt 2>&1
done
echo "T7 DONE $(date)"
