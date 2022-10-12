#!/bin/sh
SECONDS=0
gpu=0
s=0
e=1
port=2000

_term() {
  echo "Caught SIGTERM signal!"
  kill -TERM "$child" 2>/dev/null
}
trap _term SIGTERM

for i in $(seq $s $e); do
  echo "[repeat_run] starting attention_planner_testing.py script"
  echo "[repeat_run] iteration: $i"
  echo "[repeat_run] gpu_id: $gpu"
  python3 ./attention_planner_testing.py &

  child=$!
  wait "$child"
  kill -9 "$child"
  echo "[repeat_run] clearing process"
  python3 ./clear_process.py $port
  sleep 5
done
echo "Exp finished in "$SECONDS" seconds"
