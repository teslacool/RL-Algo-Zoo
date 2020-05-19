#!/bin/bash

if [ "$1" == '-h' ]
then
  echo "bash train_walker.sh 7 ppo Walker2d-v2 5 ..."
  exit
fi

export CUDA_VISIBLE_DEVICES=${1:-7}
shift
set -e
cd ..
algo=${1:-"ppo"}
shift
env=${1:-"Walker2d-v2"}
shift
maxseed=${1:-5}
shift
logpath=${algo}_${env}
logpathsuffix=`echo $* | sed -r 's/\s/_/g' | sed -r 's/-//g'`
logpath=${logpath}_${logpathsuffix}
mkdir -p logs/$logpath


maxseed=$(( maxseed - 1 ))
for seed in `seq 0 $maxseed`
do
  expname=${logpath}-s${seed}
  set -x
  python run.py --algo $algo --exp_name $expname --log_path logs/$logpath  \
    --log-freq 10 --rm_prev_log --env $env --seed $seed $@ \
    >/dev/null 2>&1 &
  pid=$!
  set +x
  PID_LIST+=" $pid"
done

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started"
echo
echo "Logging to $logpath"
wait $PID_LIST
echo
echo "All processes have completed"