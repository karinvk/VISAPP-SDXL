import pandas as pd
import os

def ap_result(valid_accuracies,avg_ap_score,avg_precision_score,avg_recall_score,csv_result_root_name):
    max_valid_acc=max(valid_accuracies)
    
    data = {
    'Experiment_setting': f"{args.evaluation_model}_{args.num_pos_original}_{args.root_pos_generated}_{args.num_pos_generated}",
    'Max_Valid_Accuracies': max_valid_acc,
    'Avg_AP_Score': avg_ap_score,
    'Avg_Precision_Score': avg_precision_score,
    'Avg_Recall_Score': avg_recall_score
    }
    df = pd.DataFrame(data)

    csv_path = os.path.join(csv_result_root)
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        df = pd.concat([existing_df, df], ignore_index=True) 
    else:
        df = df  
    df.to_csv(csv_path, index=False)
    
