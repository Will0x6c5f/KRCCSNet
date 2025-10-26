M=$1
CFG=$2
PORT=$3
bash tools/dtrain.sh $M 0.5 $CFG $PORT
bash tools/dtrain.sh $M 0.25 $CFG $PORT
bash tools/dtrain.sh $M 0.125 $CFG $PORT
bash tools/dtrain.sh $M 0.0625 $CFG $PORT
bash tools/dtrain.sh $M 0.03125 $CFG $PORT
bash tools/dtrain.sh $M 0.015625 $CFG $PORT