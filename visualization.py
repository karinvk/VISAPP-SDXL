import matplotlib.pyplot as plt
import pandas as pd


def ap_result_visual(csv_result_root_name):
    if os.path.exists(csv_result_root_name):
        df_ap = pd.read_csv(csv_path)
        
        #TODO.
        plt.figure(figsize=(10, 6))
        plt.plot(max_valid_acc.keys(), max_valid_acc.values(), marker='o')
        plt.xlabel('Percentage')
        plt.ylabel('Max Validation Accuracy')
        plt.title('Max Validation Accuracy for Each Percentage')
        plt.grid(True)
        plt.savefig('plot by percentage.png')
        plt.close()
