###
# @Author: Ethan Chen
# @Date: 2022-04-13 13:01:17
# @LastEditTime: 2022-04-13 18:02:07
# @LastEditors: Ethan Chen
# @Description:
# @FilePath: /CMPUT414/train.sh
###

for DATASET in shapenet modelnet40; do
    for LOSS_NET in DGCNN PointNet; do
        for AUGMENTION in True False; do
            python src/train.py --dataset $DATASET --loss_net $LOSS_NET --augmention $AUGMENTION --epochs 100
        done
    done
done
