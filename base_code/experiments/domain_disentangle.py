import torch
from models.base_model import DomainDisentangleModel
from domain_disentagle_losses import *

W1= 0.3
W2= 0.3
W3= 0.4
ALPHA_ENTROPY= 1
class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
         # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model =DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion = [torch.nn.CrossEntropyLoss(), NegHLoss(), torch.nn.CrossEntropyLoss(), NegHLoss(), ReconstructionLoss()]

        

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

        

    def train_iteration(self, data):
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x, 0)
        loss0= self.criterion[0](logits, y)

        logits = self.model(x, 1)
        loss1= self.criterion[1](logits)

        logits = self.model(x, 2)
        loss2= self.criterion[2](logits, y)

        logits = self.model(x, 3)
        loss3= self.criterion[3](logits)

        logits = self.model(x, 4)
        loss4= self.criterion[4](logits, y)

        
        loss= W1 * (loss0 + ALPHA_ENTROPY * loss1) +  W2 * (loss2 + ALPHA_ENTROPY * loss3) + W3 * loss4
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss += self.criterion(logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss