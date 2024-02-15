# --- imports ---
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import hw3utils
import copy
import sys
import utils
torch.multiprocessing.set_start_method('spawn', force=True)

# ---- options ----
use_gpu = torch.cuda.is_available()
if use_gpu:    
    DEVICE_ID = 'cuda' # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
else:
    DEVICE_ID = 'cpu'

LOG_DIR = 'checkpoints'
VISUALIZE = True # set True to visualize input, prediction and the output from the last batch
LOAD_CHKPT = False # set True to load the last checkpoint instead of training from scratch

# ---- utility functions -----
def get_loaders(batch_size,device):
    data_root = 'ceng483-hw3-dataset' 
    train_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'train'),device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'test'),device=device)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader, test_loader

# ---- ConvNet -----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            nn.Conv2d(8, 3, 3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            #nn.Tanh(),
        )

    def forward(self, grayscale_image):    
        x = self.conv1(grayscale_image)
        return x

def twelve_margin_error(prediction, target):
    target = (target[:,:,:].reshape(-1) / 2 + 0.5)*255
    prediction = (prediction[:,:,:].reshape(-1) / 2 + 0.5)*255

    accuracy = (np.abs(target - prediction) < 12).sum() / target.shape[0]
    return accuracy

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()  
    running_loss = 0.0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)

        # do forward
        preds = model(input)
        loss = criterion(preds, target)
        
        # do backward and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss

# validation function
def validate(model, val_loader, criterion, device, epoch):
    model.eval()  
    total_accuracy = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.to(device), target.to(device)
            
            # do forward
            pred = model(input)
            loss = criterion(pred, target)
            val_loss += loss.item()

            accuracy = twelve_margin_error(pred.cpu().numpy(), target.cpu().numpy())
            total_accuracy += accuracy
            """ 
            if (epoch+1 == 10) and VISUALIZE: 
                hw3utils.visualize_batch(input,pred,target)
            """
            

    avg_accuracy = total_accuracy / len(val_loader)
    avg_validate_loss = val_loss / len(val_loader)
    
    return avg_validate_loss, avg_accuracy

# ---- Training and Validation Code ----
train_losses = []
validate_losses = []
accuracies = []
best_accuracy = 0.0

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, max_num_epoch, validation_frequency, patience):    
    counter = 0
    best_model_state = None
    global best_accuracy

    for epoch in range(max_num_epoch):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Check validation loss if it's the right epoch, or if it's the last epoch
        if (epoch + 1) % validation_frequency == 0 or (epoch + 1) == max_num_epoch:
            
            avg_validate_loss, accuracy = validate(model, val_loader, criterion, device, epoch)
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_validate_loss:.4f}, Accuracy: {accuracy:.4f}')
            validate_losses.append(avg_validate_loss)
            
            # Early stopping logic
            if accuracy > best_accuracy + 0.0001:
                best_accuracy = accuracy
                counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                #torch.save(best_model_state, os.path.join(LOG_DIR, 'early_stop_checkpoint.pt_experiment_{}_{:.3f}'.format(best_accuracy)))
            else:
                counter += 1
                print(f'Early stopping counter: {counter}/{patience}')

            accuracies.append(best_accuracy)
            
            if counter >= patience:
                print(f'Early stopping after {epoch+1} epochs.')
                break
        else:
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss not checked')

        train_losses.append(avg_train_loss)
        

    return best_model_state

# ---- Testing Code ----
def test(test_loader, model, device):
    model.eval()
    predictions = []
    image_paths = []

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if i == 0:  # Process only the first batch of 100 images
                input = input.to(device)
                output = model(input)
                
                # Convert images to range [0, 255]
                output = ((output + 1) / 2 * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
                output = np.transpose(output, (0, 2, 3, 1))  

                predictions.extend(output)

                # Save the paths of these images
                for path, _ in test_loader.dataset.imgs[:100]:
                    image_paths.append(path)

                break  # Exit after processing the first batch

    # Save the predictions and image paths
    np.save('estimations_test.npy', np.array(predictions))
    with open('test_images.txt', 'w') as f:
        for path in image_paths:
            f.write(path + '\n')

    print("Testing completed and results saved.")


# ---- main variables -----
batch_size = 16
max_num_epoch = 100
hps = {'lr':0.05}

device = torch.device(DEVICE_ID)
print('device: ' + str(device))

model = Net().to(device=device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=hps['lr'])
train_loader, val_loader, test_loader = get_loaders(batch_size,device)

validation_frequency = 2
patience = 6

############## Main Function ##############    

# Testing
if len(sys.argv) > 1:
    if sys.argv[1] == 'test':
        if(len(sys.argv) < 3):
            print("Provide a model path to perform testing")
            exit(1)
        print('testing begins')
        
        model.load_state_dict(torch.load(sys.argv[2]))
        test(test_loader, model, device)

# Training
else:
    # Train and evaluate the model
    print('Training begins')
    best_model_state = train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, max_num_epoch, validation_frequency, patience)

    # Save the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        torch.save(best_model_state, os.path.join(LOG_DIR, 'model_checkpoint.pt_{:.3f}'.format(max(accuracies))))
    
    print('Finished Training')
    """
    print("--------------------")
    print(max(accuracies))
    print(min(validate_losses))
    print(min(train_losses))
 
    validate_epochs = range(1, len(train_losses)+1, validation_frequency)

    plt.figure(figsize=(12, 6))
    # Plot training loss for each epoch
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss', marker='o')

    # Plot validation loss for the epochs where validation was performed
    plt.plot(validate_epochs, validate_losses, label='Validation Loss', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epoch Graph')
    plt.legend()
    plt.show()

    # Plot validation accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(validate_epochs, accuracies, label='Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch Graph')
    plt.legend()
    plt.show()
    """
    
