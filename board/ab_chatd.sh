#!/bin/sh
# A/B the production daemon: whole-buffer readback (as installed) vs narrow readback.
# Each tag gets its OWN socket, and the runner waits until no e41 process is left, so a
# tag can never be served by the previous tag's daemon.  Nothing here overwrites
# /mnt/data/ernie/e41_chatd.py.
D=/mnt/data/ernie/ab
mkdir -p $D
IDS='[4,23,23,23,2969,93919,4,93963,93919,253,172,177]'
wait_clear() {
  i=0
  while [ $i -lt 30 ]; do
    n=$(ps -ef | grep -c '[e]41_chatd')
    [ "$n" = "0" ] && return 0
    sleep 2
    i=$((i + 1))
  done
  echo "WARN processes still up: $(ps -ef | grep '[e]41_chatd')" >> $D/progress.txt
  return 1
}
echo "PRE CmaFree=$(awk '/CmaFree/{print $2}' /proc/meminfo) kB Avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) kB" >> $D/progress.txt
run() {
  tag=$1; script=$2
  wait_clear
  echo "START $tag $(date -u +%H:%M:%S) script=$script" >> $D/progress.txt
  printf '{"ids":%s,"max_new":16}\n{"bye":true}\n' "$IDS" | \
    E41_SOCK=/tmp/e41_$tag.sock E41_LOCK=/tmp/e41_$tag.lock E41_LOG=$D/$tag.daemon.log E41_IDLE=1 \
    timeout 900 python3 -u $script connect > $D/$tag.jsonl 2> $D/$tag.err
  echo "END $tag rc=$? $(date -u +%H:%M:%S)" >> $D/progress.txt
  echo "POST $tag CmaFree=$(awk '/CmaFree/{print $2}' /proc/meminfo) kB" >> $D/progress.txt
}
run OLD /mnt/data/ernie/e41_chatd.py
run NEW /mnt/data/ernie/e41_chatd_narrow.py
echo "ALLDONE $(date -u +%H:%M:%S)" >> $D/progress.txt
