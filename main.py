#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data.ksdd2 import KolektorSDD2
from model.resnet import ResNet
from train.trainer import train
import os
import pandas as pd

def main(args):
    # Dataset.
    print('Loading KolektorSDD2 training set...')
    train_data = KolektorSDD2(dataroot=args.dataset_path, split='train', scale='half', debug=False, positive_percentage=args.pos_percentage)
    print('Number of samples:', len(train_data))

    print('Loading KolektorSDD2 test set...')
    test_data = KolektorSDD2(dataroot=args.dataset_path, split='test', scale='half', debug=False)
    print('Number of samples:', len(test_data))
    
    # DataLoader.
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=4)

    # Train Setting.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=ResNet.resnet18(num_classes=2, input_img_size=(704, 256),pre_trained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) #optimizer = torch.optim.Adam(params, lr=0.0001)

    # Train and save model
    train_losses, valid_accuracies, avg_ap_score, avg_precision_score, avg_recall_score=train(train_loader, test_loader, model, optimizer, criterion, args.epochs, device, target_accuracy=None, model_save_path='./saved')
    df = pd.DataFrame({'train_loss': train_losses, 'valid_accuracy': valid_accuracies})
    file_name = f"results_{args.pos_percentage}.csv"
    df.to_csv(file_name, index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the KolektorSDD2 dataset')
    parser.add_argument('--dataset_path', type=str, default='.', help='Path to the KolektorSDD2 dataset')
    parser.add_argument('--epochs',type=int,default=10,help='epochs, default 10')
    parser.add_argument('--lr',type=float, default=0.001,help='learning rate, default 0.001')
    parser.add_argument('--num_pos_original', type=int, default=246, help='keep certain number of defect images coming from the original dataset when training, default is keeping all 246 positive images')
    parser.add_argument('--num_pos_generated', type=int, default=0, help='add certain number of synthesized defect images, default is 0')
    parser.add_argument('--root_pos_generated', type=str, default=None, help='synthesized defect images folder')
    args = parser.parse_args()
    main(args)
