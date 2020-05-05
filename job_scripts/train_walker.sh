#!/bin/bash
export CUDA_VISIBLE_DEVICES=${1:-7}
set -e
cd ..

env=${2:-"Walker2d-v2"}
loggpath=logs/$env
mkdir -p $loggpath

maxseed=${3:-5}
for seed in `seq 1 $maxseed`
do
  expname=${env}-s${seed}
  set -x
  python run.py --exp_name $expname --log_path $loggpath  \
    --log-freq 10 --rm_prev_log --env $env --seed $seed  \
    >/dev/null 2>&1 & pid=$!
  set +x
  PID_LIST+=" $pid"
done

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started"
echo
echo "Logging to $loggpath"
wait $PID_LIST
echo
echo "All processes have completed"