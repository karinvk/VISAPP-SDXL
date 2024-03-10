import matplotlib.pyplot as plt
import pandas as pd

def train_vis_by_epoch(train_losses,valid_accuracies,test_name):
    df = pd.DataFrame({'train_loss': train_losses, 'valid_accuracy': valid_accuracies})
    #file_name = f"epoch_{args.evaluation_model}_{args.num_pos_original}_{args.root_pos_generated}_{args.num_pos_generated}.csv"
    #df.to_csv(file_name, index=False)
    plt.plot(train_losses, label=f'Train Loss {test_name}')
    plt.plot(valid_accuracies, label=f'Val Accuracy {test_name}') 
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training Loss and Validation Accuracy')
    plt.grid(True)
    plt.savefig(f'epoch_{test_name}.png')
    plt.close()

def ap_result_visual(csv_result_root_name):
    

    plt.figure(figsize=(10, 6))
    plt.plot(max_valid_acc.keys(), max_valid_acc.values(), marker='o')
    plt.xlabel('Percentage')
    plt.ylabel('Max Validation Accuracy')
    plt.title('Max Validation Accuracy for Each Percentage')
    plt.grid(True)
    plt.savefig('plot by percentage.png')
    plt.close()
