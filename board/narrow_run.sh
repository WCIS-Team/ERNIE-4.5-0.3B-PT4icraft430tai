#!/bin/sh
# Narrow-read validation + per-op profile rungs.  One rung per process: apply() is
# not reentrant, and two live sessions would fight over the same CMA/PL DDR.
cd /mnt/data/ernie || exit 3
mkdir -p /mnt/data/ernie/narrow
P=/mnt/data/ernie/narrow/progress.txt
D=/mnt/data/ernie/narrow
say() { echo "$1 $(date -u +%H:%M:%S)" >> "$P"; }

rung() {
  tag=$1
  case $tag in
    W)  set -- env E41N_CFG=W E41N_READ=whole E41N_STEPS=12 E41N_AUDIT=0 \
           E41N_OUT=$D/W.txt E41N_OUTJSON=$D/W.json \
           python3 -u /mnt/data/ernie/e41_narrow.py ;;
    N)  set -- env E41N_CFG=N E41N_READ=narrow E41N_STEPS=12 E41N_AUDIT=1 \
           E41N_OUT=$D/N.txt E41N_OUTJSON=$D/N.json \
           python3 -u /mnt/data/ernie/e41_narrow.py ;;
    N0) set -- env E41N_CFG=N0 E41N_READ=narrow E41N_STEPS=12 E41N_AUDIT=0 \
           E41N_OUT=$D/N0.txt E41N_OUTJSON=$D/N0.json \
           python3 -u /mnt/data/ernie/e41_narrow.py ;;
    PJ) set -- env E41PROF=1 E41KV=bindonly E41PAD=host E41DEAD=host E41RESET=1 \
           E41P_CFG=PJ E41P_STEPS=8 E41P_OUT=$D/PJ.txt E41P_OUTJSON=$D/PJ.json \
           python3 -u /mnt/data/ernie/e41_kvres.py ;;
    PB) set -- env E41PROF=1 E41KV=inplace E41PAD=dev E41DEAD=host E41RESET=1 \
           E41P_CFG=PB E41P_STEPS=8 E41P_OUT=$D/PB.txt E41P_OUTJSON=$D/PB.json \
           python3 -u /mnt/data/ernie/e41_kvres.py ;;
    *)  echo "BADRUNG $tag" >> "$P"; return 1 ;;
  esac
  say "START $tag"
  "$@" > "$D/$tag.stdout" 2>&1
  say "END $tag rc=$?"
}

for r in $RUNGS; do rung "$r"; done
say ALLDONE
