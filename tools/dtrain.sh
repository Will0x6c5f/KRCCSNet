M=$1
SR=$2
CFG=$3
PORT=$4
PORT=${PORT:-29377}
echo $PORT
accelerate launch --main_process_port $PORT --config_file $CFG dist_train.py --model $M --sensing-rate $SR \
 --epochs 200 --batch-size 4 --block-size 32 \
 --image-size 96 --lr 1e-3 | tee -a log/dist_$M$SR.log