#!/usr/bin/env bash
set -x
set -e
cd ..

ALGO=sac
ENV=Walker2d-v2
SEEDS=2
CUDA=0

POSITIONAL=()

while [[ $# -gt 0 ]]; do
    key=$1
    case $key in
      -a|--algo)
        ALGO=$2
        shift 2
        ;;
      -e|--env)
        ENV=$2
        shift 2
        ;;
      -ss|--seeds)
        SEEDS=$2
        shift 2
        ;;
      -c|--cuda)
        CUDA=$2
        shift 2
        ;;
      *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$CUDA
logpath=${ALGO}_${ENV}
suffix=`echo ${POSITIONAL[*]} | sed -r 's/-//g' | sed -r 's/\s+/-/g'`
if [ -n "$suffix" ]
then
  logpath=${logpath}-${suffix}
else
  logpath=$logpath
fi
mkdir -p logs/$logpath

maxseed=$(( SEEDS - 1 ))
for seed in `seq 0 $maxseed`
do
  expname=${logpath}-s${seed}
  python run.py --algo $ALGO --env $ENV --log_path logs/$logpath \
    --exp_name $expname --log-freq 10 --seed $seed ${POSITIONAL[@]} \
    >/dev/null 2>&1 &
  pid=$!
  PID_LIST+=" $pid"
done

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started"
echo
echo "Logging to $logpath"
wait $PID_LIST
echo
echo "All processes have completed"
