'''
Author: Ethan Chen
Date: 2022-04-14 22:38:52
LastEditTime: 2022-04-16 23:43:16
LastEditors: Ethan Chen
Description: 
FilePath: /CMPUT414/src/evaluation.py
'''
import argparse
import csv
import os
import sys
from loss import DGCNN
from metrics.metric import l1_cd, l2_cd, f_score
from data import *
from datetime import datetime

from model import PCN
EVALUATE_RESULT_PATH = '/home/ethan/Code/Project/CMPUT414/results/New'


def write_result(name, path, shape, result_original, result_translated, result_rotated):
    with open(os.path.join(path, 'result.csv'), 'a') as f:
        csv_wite = csv.writer(f)
        csv_wite.writerow([name, SHAPE_NAME[shape], result_original[0], result_original[1], result_original[2],
                          result_translated[0], result_translated[1], result_translated[2],
                          result_rotated[0], result_rotated[1], result_rotated[2]])


def init_result(path):
    with open(os.path.join(path, 'result.csv'), 'w') as f:
        csv_wite = csv.writer(f)
        csv_wite.writerow(['model', 'shape', 'original_l1_cd', 'original_l2_cd', 'original_f1', 'translated_l1_cd',
                          'translated_l2_cd', 'translated_f1', 'rotated_l1_cd', 'rotated_l2_cd', 'rotated_f1'])


def evaluate_single(model, pc, args):
    _, pred = model(pc[2].to(args.device))
    f1, l1cd, l2cd = 0, 0, 0

    l1cd += l1_cd(pred, pc[0].to(args.device)).item()
    l2cd += l2_cd(pred, pc[0].to(args.device)).item()

    # print(len(pc[0]))
    for i in range(len(pc[0])):
        single_pc = pc[0][i].detach().cpu().numpy()
        single_pred = pred[i].detach().cpu().numpy()
        f1 += f_score(single_pred, single_pc)
    return (l1cd, l2cd, f1)


def evaluate_single_shape(shape, model, args):
    modelnet40_test = ModelNet40(num_points=2048, data_path=MODELNET40_TEST, shape=shape, augmentation=True, severity=5)
    test_loader = DataLoader(modelnet40_test, num_workers=8, batch_size=32, shuffle=True)

    # evaluate model

    original_l1_cd, original_l2_cd, original_f1 = 0, 0, 0
    translated_l1_cd, translated_l2_cd, translated_f1 = 0, 0, 0
    rotated_l1_cd, rotated_l2_cd, rotated_f1 = 0, 0, 0
    length = len(test_loader.dataset)
    model.eval()

    with torch.no_grad():
        for _, data in enumerate(test_loader):
            original, translated, rotated, _ = data

            # original
            result = evaluate_single(model, original, args)
            original_l1_cd += result[0]
            original_l2_cd += result[1]
            original_f1 += result[2]

            # translated
            result = evaluate_single(model, translated, args)
            translated_l1_cd += result[0]
            translated_l2_cd += result[1]
            translated_f1 += result[2]

            # rotated
            result = evaluate_single(model, rotated, args)
            rotated_l1_cd += result[0]
            rotated_l2_cd += result[1]
            rotated_f1 += result[2]

    ave_original_l1_cd = original_l1_cd / length
    ave_original_l2_cd = original_l2_cd / length
    ave_original_f1 = original_f1 / length

    ave_translated_l1_cd = translated_l1_cd / length
    ave_translated_l2_cd = translated_l2_cd / length
    ave_translated_f1 = translated_f1 / length

    ave_rotated_l1_cd = rotated_l1_cd / length
    ave_rotated_l2_cd = rotated_l2_cd / length
    ave_rotated_f1 = rotated_f1 / length

    result_original = (ave_original_l1_cd, ave_original_l2_cd, ave_original_f1)
    result_translated = (ave_translated_l1_cd, ave_translated_l2_cd, ave_translated_f1)
    result_rotated = (ave_rotated_l1_cd, ave_rotated_l2_cd, ave_rotated_f1)

    return result_original, result_translated, result_rotated


def evaluate_model(completion_model, args):

    path = os.path.join(EVALUATE_RESULT_PATH, args.exp)
    if os.path.exists(path):
        path = os.path.join(path, datetime.now().strftime('%m-%d-%H-%M-%S'))
    os.makedirs(path)
    print("Evaluate model: {}".format(path))
    init_result(path)

    completion_model.eval()
    if args.dataset == 'modelnet40':
        for shape in range(40):
            print("Shape: {}".format(SHAPE_NAME[shape]))
            result_original, result_translated, result_rotated = evaluate_single_shape(shape, completion_model, args)
            write_result(args.exp, path, shape, result_original, result_translated, result_rotated)
    elif args.dataset == 'shapenet':
        # TODO: evaluate shapenet
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='PCN_CD_alpha_PN(AU)_S5', help='experiment name')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='device for evaluation')

    args = parser.parse_args()

    # TODO: load model
    completion_model = PCN(2048).to(args.device)
    completion_model.load_state_dict(torch.load(
        '/home/ethan/Code/Project/CMPUT414/model/PNLoss_alpha_PN(AU)_S5/completion_model_380.pth'))
    evaluate_model(completion_model, args)
