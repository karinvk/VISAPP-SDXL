import torch
from tqdm import tqdm 
import os
from torchmetrics import AveragePrecision, Precision, Recall
from torch import nn
from utils.metrics import get_metrics
import numpy as np

def train(trainloader, testloader, model, optimizer, criterion, epochs, device, target_accuracy=None, model_save_name=''):
    best_ap,best_precision,best_recall = 0.0, 0.0, 0.0
    train_accuracies = []  
    train_losses = []  
    valid_accuracies = []  
    

    for epoch in range(epochs):  
        
        model.train()
        running_loss = 0.0
        running_correct = 0   
        total_train = 0   
        epoch_predictions, epoch_ground_truths = [], []

        train_bar=tqdm(trainloader,position=0,leave=True) 
        for i, data in enumerate(train_bar):
            images, labels, mask, info = data 
            optimizer.zero_grad() 
            preds_scores = model(images.to(device)) 
            _, preds_class = torch.max(preds_scores, 1) #preds_class = torch.argmax(preds_scores, dim=-1)
            loss = criterion(preds_scores, labels.to(device))
            loss.backward() 
            optimizer.step()

            running_loss += loss.item() 
            running_correct += (preds_class == labels.to(device)).sum().item() #torch.sum(preds_class == labels)
            total_train += labels.size(0)
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs,
                                                                    loss)

        # print statistics
        train_accuracy = running_correct / total_train 
        train_accuracies.append(train_accuracy)
        train_loss = running_loss / len(trainloader) 
        train_losses.append(train_loss)
        

        if target_accuracy != None:
            if train_accuracy > target_accuracy:
                print("Early Stopping")
                break

        model.eval()
        correct_valid = 0
        total_valid = 0
        acc = 0.0  
        with torch.no_grad():
            val_bar = tqdm(testloader)
            for val_data in val_bar:
                val_images, val_labels, val_mask, val_info = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                preds_scores = model(val_images) 
                # loss = loss_function(outputs, test_labels)
                preds_prob , preds_class = torch.max(preds_scores, 1)  # torch.max(preds_scores, dim=1)[1]
                correct_valid += torch.eq(preds_class, val_labels).sum().item()
                total_valid += val_labels.size(0)
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                            epochs)
                
                prediction = torch.sigmoid(preds_scores)[:,1]        
                epoch_predictions.extend(prediction.tolist())
                epoch_ground_truths.extend(val_labels.tolist())
  
                
        valid_accuracy = correct_valid / total_valid
        valid_accuracies.append(valid_accuracy) # valid_losses.append(loss.item())
        
        epoch_metrics = get_metrics(np.array(epoch_ground_truths), np.array(epoch_predictions))

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                (epoch + 1, train_loss, valid_accuracy)) 

        if epoch_metrics['ap_score'] > best_ap:
            best_ap = epoch_metrics['ap_score']
            best_precision = epoch_metrics['precision_score']
            best_recall = epoch_metrics['recall_score']
            ap_auc=epoch_metrics['ap_score_auc']

            model_save_path = os.path.join('./saved/model/' + model_save_name)
            if not os.path.exists('./saved/model/'):
                os.makedirs('./saved/model/')
            torch.save(model.state_dict(), model_save_path)

    print('Finished Training')
    
    return train_losses, valid_accuracies, best_ap, best_precision, best_recall, ap_auc
