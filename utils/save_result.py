import pandas as pd
import os
import matplotlib.pyplot as plt

def train_vis_by_epoch(train_losses,valid_accuracies,experiment_name):
    df_epoch = pd.DataFrame({'train_loss': train_losses, 'valid_accuracy': valid_accuracies})
    file_name = f"epoch_{experiment_name}.csv"
    df_epoch.to_csv(file_name, index=False)
    
    plt.plot(train_losses, label=f'Train Loss {experiment_name}')
    plt.plot(valid_accuracies, label=f'Val Accuracy {experiment_name}') 
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training Loss and Validation Accuracy')
    plt.grid(True)
    plt.savefig(f'epoch_{experiment_name}.png')
    plt.close()
    
def ap_result(avg_ap_score,avg_ap_score_auc,avg_precision_score,avg_recall_score,experiment_name,csv_result_root):
    #max_valid_acc=max(valid_accuracies)
    
    data = {
    'Experiment_setting': experiment_name,
    'Avg_AP_Score': avg_ap_score,
    'Avg_AP_Score_AUC':avg_ap_score_auc,
    'Avg_Precision_Score': avg_precision_score,
    'Avg_Recall_Score': avg_recall_score
    }
    df = pd.DataFrame([data])

    csv_path = os.path.join(csv_result_root)
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        df = pd.concat([existing_df, df], ignore_index=True) 
    else:
        df = df  
    df.to_csv(csv_path, index=False)
    
