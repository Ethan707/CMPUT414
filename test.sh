###
# @Author: Ethan Chen
# @Date: 2022-04-14 12:09:23
# @LastEditTime: 2022-04-14 12:09:23
# @LastEditors: Ethan Chen
# @Description:
# @FilePath: /CMPUT414/test.sh
###

python src/train.py --dataset $DATASET --loss_net $LOSS_NET --augmention $AUGMENTION --epochs 1
