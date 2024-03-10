import pandas as pd
import os

data = {
    'Experiment_setting': f"{args.evaluation_model}_{args.num_pos_original}_{args.root_pos_generated}_{args.num_pos_generated}",
    'Min_train_Losses': train_losses,
    'Max_Valid_Accuracies': valid_accuracies,
    'Avg_AP_Score': avg_ap_score,
    'Avg_Precision_Score': avg_precision_score,
    'Avg_Recall_Score': avg_recall_score
}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 检查csv文件是否存在，如果存在则加载，否则创建一个新的DataFrame
csv_path = os.path.join(args.csv_result_root, 'results.csv')
if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path)
    df = pd.concat([existing_df, df], ignore_index=True)  # 将新数据添加到已有数据后面
else:
    df = df  # 如果不存在，则使用新的DataFrame

# 将DataFrame写入CSV文件
df.to_csv(csv_path, index=False)
