#!/bin/sh
# 逐 token oracle 回放：同一段已落盘提示（7 token＋padfix 双喂＝8 次输入，63 步）
# 在 8 档上各跑一遍，步进 JSON 用来跟 e41_long.json 逐 step 对 logits 极值。
# 一档一进程、跑完就退，CMA 归还后再起下一档。
D=/mnt/data/ernie/oracle
mkdir -p $D
P=$D/progress.txt
: > $P
PFX=94515,94515,94036,31282,5502,44284,3458,93977

run() {
  echo "START $1 $(date +%H:%M:%S)" >> $P
  env E41P_CFG=$1 E41P_OCM=$2 E41P_BEST=$3 E41P_ETM=$4 E41P_MERGE=$5 \
      E41_PADFIX=1 E41P_STEPS=63 E41P_PREFIX=$PFX \
      E41P_OUT=$D/$1.txt E41P_OUTJSON=$D/$1.json \
      timeout 900 python3 -u /mnt/data/ernie/e41_ocm.py > $D/$1.log 2>&1
  echo "END $1 rc=$? $(date +%H:%M:%S)" >> $P
}

run ocm0     0 0 on  on
run ocm1     1 0 on  on
run ocm2     2 0 on  on
run ocm3     3 0 on  on
run best     4 1 on  on
run etmoff   1 0 off on
run mergeoff 1 0 on  off
run bothoff  1 0 off off
echo "ALLDONE $(date +%H:%M:%S)" >> $P
