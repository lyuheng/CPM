export PYTHONUNBUFFERED="True"
LOG="./logs/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"
python train.py  > $LOG