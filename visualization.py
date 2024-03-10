import matplotlib.pyplot as plt
import pandas as pd

def train_vis_by_epoch(train_losses,valid_accuracies,):
    df = pd.DataFrame({'train_loss': train_losses, 'valid_accuracy': valid_accuracies})
    #file_name = f"epoch_{args.evaluation_model}_{args.num_pos_original}_{args.root_pos_generated}_{args.num_pos_generated}.csv"
    #df.to_csv(file_name, index=False)

def ap_result_visual():
    

    


max_valid_acc={}
plt.figure(figsize=(19, 8))
for i in range(11):
    print('percentage:',i)
    df=pd.read_csv('/home/zliu/github/sd_utilities/in-and-out-VISAPP/results_{:.1f}.csv'.format(i*0.1))
    train_loss = df['train_loss']
    valid_accuracy = df['valid_accuracy']
    plt.plot(train_loss, label=f'Train Loss ({i * 10}%)')
    plt.plot(valid_accuracy, label=f'Val Accuracy ({i * 10}%)') 
    max_valid_acc[f'{i * 10}%']=max(valid_accuracy)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  
plt.xlabel('Epoch')
plt.ylabel('Loss / Accuracy')
plt.title('Training Loss and Validation Accuracy')
plt.grid(True)
plt.savefig('plot by epoch.png')
#plt.figure(figsize=(10, 6))
# Close the plot to release memory
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(max_valid_acc.keys(), max_valid_acc.values(), marker='o')
plt.xlabel('Percentage')
plt.ylabel('Max Validation Accuracy')
plt.title('Max Validation Accuracy for Each Percentage')
plt.grid(True)
plt.savefig('plot by percentage.png')
plt.close()
