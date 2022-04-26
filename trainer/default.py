from logger.default import Logger
import torch
import torchvision
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



class Trainer:
    def __init__(self, config, device, model, trainval_dataloaders, criterion, optimizer, lr_scheduler, logger):
        self.config = config
        self.device = device
        self.model = model.to(self.device)
        self.logger = logger
        self.train_dataloader = trainval_dataloaders['train']
        self.val_dataloader = trainval_dataloaders['val']
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.iter = 1
        
        self.num_of_epochs = self.config['NUM_OF_EPOCHS']

    def train_epoch(self):
        self.model.train()
        
        for inputs, targets in self.train_dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()

            self.logger.add_scalar('train_loss', loss.item(), global_step = self.iter)
            self.logger.log_info('Iter {} train loss: {}'.format(self.iter, loss.item()))

            self.optimizer.step()
            self.optimizer.zero_grad()
            loss = None
            
            self.iter += 1

    def valid_epoch(self):
        self.model.eval()

        total = 0
        correct = 0
        running_loss = []

        class_names = [
            'White', 
            'Black', 
            'Latino_Hispanic', 
            'East Asian', 
            'Indian', 
            'Middle Eastern'
        ]
        nb_classes = len(class_names)

        confusion_matrix = np.zeros((nb_classes, nb_classes))

        with torch.no_grad():
            for inputs, targets in self.val_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)

                correct += (preds == targets).sum()
                total += targets.shape[0]
                loss = self.criterion(outputs, targets)
                running_loss.append(loss.mean().item())

                for t, p in zip(targets.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        plt.figure(figsize=(15,10))

        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        running_loss = sum(running_loss) / len(running_loss)

        return correct / total, running_loss, plt


    def train(self):
        for epoch in range(self.num_of_epochs):
            self.logger.log_info('Starting training of epoch {}.'.format(epoch))
            self.train_epoch()
            accuracy, loss, plt = self.valid_epoch()
            self.logger.log_info('Epoch {}: Accuracy: {} Loss: {}'.format(epoch, accuracy, loss))
            plt.savefig('out/figure{}.png'.format(epoch + 1))
            torch.save(self.model.state_dict(), 'out/model{}.pt'.format(epoch + 1))
