#!/bin/bash
export CUDA_VISIBLE_DEVICES=${1:-7}
set -e
cd ..
algo=${2:-"ppo"}
env=${3:-"Walker2d-v2"}
loggpath=logs/${algo}
mkdir -p $loggpath

maxseed=${4:-5}
maxseed=$(( maxseed - 1 ))
for seed in `seq 0 $maxseed`
do
  expname=${env}-s${seed}
  set -x
  python run.py --algo $algo --exp_name $expname --log_path $loggpath  \
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