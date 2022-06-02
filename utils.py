import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import time

import matplotlib.pyplot as plt

def sample_from_data(dataset, sample_size = 4, show_gray = False):
    """Function to sample from dataset

    Args:
        dataset (_type_): _description_
        sample_size (int, optional): _description_. Defaults to 4.
        show_gray (bool, optional): _description_. Defaults to False.
    """
    pass

def modelSummary(model, verbose=False):
    if verbose:
        print(model)
    
    total_parameters = 0
        
    for name, param in model.named_parameters():
        num_params = param.size()[0]
        total_parameters += num_params
        if verbose:
            print(f"Layer: {name}")
            print(f"\tNumber of parameters: {num_params}")
            print(f"\tShape: {param.shape}")
    
    if total_parameters > 1e5:
        print(f"Total number of parameters: {total_parameters/1e6:.2f}M")
    else:
        print(f"Total number of parameters: {total_parameters/1e3:.2f}K") 

def train_epoch(model: nn.Module, device: torch.device, train_dataloader: DataLoader, training_params: dict, metrics: dict):
    """_summary_

    Args:
        model (nn.Module): Model to be trained by
        device (str): device to be trained on
        train_dataloader (nn.data.DataLoader): Dataloader object to load batches of dataset
        training_params (dict): Dictionary of training parameters containing "batch_size", "loss_function"
                                "optimizer".
        metrics (dict): Dictionary of functional methods that would compute the metric value

    Returns:
        run_results (dict): Dictionary of metrics computed for the epoch
    """
    BATCH_SIZE = training_params["batch_size"]
    LOSS_FUNCTION = training_params["loss_function"]
    OPTIMIZER = training_params["optimizer"]
    
    model = model.to(device)
    model.train()
    
    # Dictionary holding result of this epoch
    run_results = dict()
    for metric in metrics:
        run_results[metric] = 0.0
    run_results["loss"] = 0.0
    
    # Iterate over batches
    num_batches = 0
    for x, target in train_dataloader:
        num_batches += 1

        # Move tensors to device
        input = x.to(device)
        
        # Forward pass
        output = model(input)
        
        # Compute loss
        loss = LOSS_FUNCTION(output, target)
        
        # Backward pass
        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        
        # Update metrics
        run_results["loss"] += loss.detach().item()
        for key, func in metrics.items():
            run_results[key] += func(output, target).detach().item()
            
        # Clean up memory
        del loss
        del input
        del output
        
    for key in run_results:
        run_results[key] /= num_batches
    
    return run_results

def evaluate_epoch(model: nn.Module, device: torch.device, validation_dataloader: DataLoader, training_params: dict, metrics: dict):
    """_summary_

    Args:
        model (nn.Module): model to evaluate
        device (str): device to evaluate on
        validation_dataloader (DataLoader): DataLoader for evaluation
        training_params (dict): Dictionary of training parameters containing "batch_size", "loss_function"
                                "optimizer".
        metrics (dict): Dictionary of functional methods that would compute the metric value

    Returns:
        run_results (dict): Dictionary of metrics computed for the epoch
    """
    LOSS_FUNCTION = training_params["loss_function"]
    
    model = model.to(device)
    
    # Dictionary holding result of this epoch
    run_results = dict()
    for metric in metrics:
        run_results[metric] = 0.0
    run_results["loss"] = 0.0
    
    # Iterate over batches
    with torch.no_grad():
        model.eval()
        num_batches = 0
        
        for x, target in validation_dataloader:
            num_batches += 1
            
            # Move tensors to device
            input = x.to(device)
            
            # Forward pass
            output = model(input)
            
            # Compute loss
            loss = LOSS_FUNCTION(output, target)
            
            # Update metrics
            run_results["loss"] += loss.detach().item()
            for key, func in metrics.items():
                run_results[key] += func(output, target).detach().item()
                
            # Clean up memory
            del loss
            del input
            del output
                
    for key in run_results:
        run_results[key] /= num_batches
        
    return run_results

def train_evaluate(model: nn.Module, device: torch.device, train_dataset: Dataset, validation_dataset: Dataset, training_params: dict, metrics: dict):
    """Function to train a model and provide statistics during training

    Args:
        model (nn.Module): Model to be trained
        device (torch.device): Device to be trained on
        train_dataset (DataLoader): Dataset to be trained on
        validation_dataset (DataLoader): Dataset to be evaluated on
        training_params (dict): Dictionary of training parameters containing "num_epochs", "batch_size", "loss_function",
                                                                             "save_path", "optimizer"
        metrics (dict): Dictionary of functional methods that would compute the metric value

    Returns:
        _type_: _description_
    """
    NUM_EPOCHS = training_params["num_epochs"]
    BATCH_SIZE = training_params["batch_size"]
    SAVE_PATH = training_params["save_path"]
    SAMPLE_SIZE = 10
    PLOT_EVERY = 1
    
    # Initialize metrics
    train_results = dict()
    train_results['loss'] = []
    evaluation_results = dict()
    evaluation_results['loss'] = []
    
    for metric in metrics:
        train_results[metric] = []
        evaluation_results[metric] = []
    
    # Create Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(NUM_EPOCHS):
        start = time.time()
        
        print(f"======== Epoch {epoch+1}/{NUM_EPOCHS} ========")

        # Train Model
        print("Training ... ")
        epoch_train_results = train_epoch(model, device, train_dataloader, training_params, metrics)
        

        # Evaluate Model
        print("Evaluating ... ")
        epoch_evaluation_results = evaluate_epoch(model, device, validation_dataloader, training_params, metrics)
        
        for metric in metrics:
            train_results[metric].append(epoch_train_results[metric])
            evaluation_results[metric].append(epoch_evaluation_results[metric])
            
        
        # Print results of epoch
        print(f"Completed Epoch {epoch+1}/{NUM_EPOCHS} in {(time.time() - start):.2f}s")
        print(f"Train Loss: {epoch_train_results['loss']:.4f} \t Validation Loss: {epoch_evaluation_results['loss']:.4f}")
        
        # # Plot results
        # if epoch % PLOT_EVERY = 0:
        #     batch = next(iter(validation_dataloader))
            
        #     model.eval()
        #     ouputs = model(batch[0].to(device)).detach().cpu()
            
        #     fig, ax = plt.subplots(2, SAMPLE_SIZE, figsize=(SAMPLE_SIZE * 5,15))
        #     for i in range(SAMPLE_SIZE):
        #         image = batch[0][i].detach().cpu()
        #         output = ouputs[i]
                
        #         ax[0][i].imshow(image.reshape(28,28))
        #         ax[1][i].imshow(output.reshape(28,28))
            
        #     plt.savefig(f"{SAVE_PATH}_epoch{epoch + 1}.png")
        #     plt.close()
        
        # # Save model
        # SAVE = f"{SAVE_PATH}_epoch{epoch + 1}.pt"
        # torch.save(model.state_dict(), SAVE)
           
    return train_results, evaluation_results


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64 , 3, stride = 2)
        self.conv2 = nn.Conv2d(64, 256, 3, stride = 2)
        self.batch2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, 512, 3, stride = 2)
        self.linear1 = nn.Linear(2*2*512, 1024)
        self.muLinear = nn.Linear(1024, latent_dims)
        self.sigmaLinear = nn.Linear(1024, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.muLinear(x)
        sigma = self.sigmaLinear(x)
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
if __name__ == '__main__':
    model = VariationalEncoder(latent_dims=2)
