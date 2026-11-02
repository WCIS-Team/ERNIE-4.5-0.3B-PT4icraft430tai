#!/bin/sh
# E41 ocm/etm/merge 单变量扫描：一档一个进程，跑完就退，CMA 归还后再起下一档。
# 末了复跑一次现役档（ocm1_recheck）——不这样就没法判断档间差是不是噪声。
D=/mnt/data/ernie/ocm
mkdir -p $D
P=$D/progress.txt
: > $P

run() {
  cfg=$1
  shift
  echo "START $cfg $(date -u +%H:%M:%S)" >> $P
  env E41P_CFG=$cfg E41P_OUT=$D/$cfg.txt "$@" \
      timeout 400 python3 -u /mnt/data/ernie/e41_ocm.py > $D/$cfg.stdout 2>&1
  echo "END $cfg rc=$? $(date -u +%H:%M:%S)" >> $P
}

run ocm0        E41P_OCM=0
run ocm1        E41P_OCM=1
run ocm2        E41P_OCM=2
run ocm3        E41P_OCM=3
run best        E41P_OCM=4 E41P_BEST=1
run etmoff      E41P_OCM=1 E41P_ETM=off
run mergeoff    E41P_OCM=1 E41P_MERGE=off
run bothoff     E41P_OCM=1 E41P_ETM=off E41P_MERGE=off
run ocm1_recheck E41P_OCM=1
echo "ALLDONE $(date -u +%H:%M:%S)" >> $P
