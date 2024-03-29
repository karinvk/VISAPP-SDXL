#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data.ksdd2 import KolektorSDD2
from model.resnet import ResNet18,ResNet50
from train.trainer import train
import os
import pandas as pd
from utils.save_result import ap_result, train_vis_by_epoch

def main(args):
    # Seed.
    torch.manual_seed(123)

    # Dataset.
    print('Loading KolektorSDD2 training set...')
    train_data = KolektorSDD2(dataroot=args.dataset_path, split='train', scale='half', debug=False, num_pos_original=args.num_pos_original, num_pos_generated=args.num_pos_generated, root_pos_generated=args.root_pos_generated)
    print('Number of samples:', len(train_data))

    print('Loading KolektorSDD2 test set...')
    test_data = KolektorSDD2(dataroot=args.dataset_path, split='test', scale='half', debug=False)
    print('Number of samples:', len(test_data))
    
    # DataLoader.
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=4)

    # Train Setting.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.evaluation_model == 'ResNet18':
        model = ResNet18(num_classes=2, input_img_size=(704, 256)).to(device)
    elif args.evaluation_model == 'ResNet50':
        model = ResNet50(num_classes=2, input_img_size=(704, 256)).to(device)
    else:
        raise ValueError("Invalid evaluation_model value. Choose from 'ResNet18' or 'ResNet50'")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) #optimizer = torch.optim.Adam(params, lr=0.0001)

    # Train and save model
    experiment_name=f"{args.evaluation_model}_{args.num_pos_original}_{args.root_pos_generated}_{args.num_pos_generated}"
    train_losses, valid_accuracies, avg_ap_score, avg_ap_score_auc, avg_precision_score, avg_recall_score=train(train_loader, test_loader, model, optimizer, criterion, args.epochs, device, target_accuracy=None, model_save_name=experiment_name)
    ap_result(avg_ap_score, avg_ap_score_auc, avg_precision_score, avg_recall_score, experiment_name, args.csv_result_root_name)
    train_vis_by_epoch(train_losses,valid_accuracies,experiment_name)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the KolektorSDD2 dataset')
    parser.add_argument('--epochs',type=int,default=50,help='default 50')
    parser.add_argument('--momentum',type=float,default=0,help='default 0')
    parser.add_argument('--lr',type=float, default=0.01,help='learning rate, default 0.01')
    parser.add_argument('--batch',type=float, default=5,help='batch size, default 5')
    parser.add_argument('--evaluation_model', type=str, default='ResNet50', help='pre-trained model used for evaluation, default is ResNet18')
    parser.add_argument('--csv_result_root_name', type=str, default='./saved/result.csv', help='default saved/result.csv')

    parser.add_argument('--dataset_path', type=str, default='./data/input/', help='Path to the KolektorSDD2 dataset')
    parser.add_argument('--num_pos_original', type=int, default=246, help='keep certain number of defect images coming from the original dataset when training, default is keeping all 246 positive images')
    parser.add_argument('--num_pos_generated', type=int, default=0, help='add certain number of synthesized defect images, default is 0')
    parser.add_argument('--root_pos_generated', type=str, default=None, help='synthesized defect images sub-folder name under generated_imgs')
    
    args = parser.parse_args()
    main(args)
