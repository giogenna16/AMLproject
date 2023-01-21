import torch
from models.base_model import BaselineModel


class BaselineExperiment: # See point 1. of the project
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = BaselineModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()

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

    def train_iteration(self, data, loss_acc_logger):
        if self.opt['domain_generalization']:
            x, y, _ = data  # the domain label in the baseline is useless
        else:
            x, y = data

        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_acc_logger['loss_log_train'] += loss
        loss_acc_logger['train_counter'] += 1

        return loss.item()

    def validate(self, loader, loss_acc_logger, test=False):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            if self.opt['domain_generalization']:
                for x, y, _ in loader: #the domain label in the baseline is useless
                    x = x.to(self.device)
                    y = y.to(self.device)

                    logits = self.model(x)
                    loss += self.criterion(logits, y)
                    pred = torch.argmax(logits, dim=-1)

                    accuracy += (pred == y).sum().item()
                    count += x.size(0)
            else:
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

        loss_acc_logger['loss_log_val'] += mean_loss
        loss_acc_logger['val_counter'] += 1        

        self.model.train()
        return mean_accuracy, mean_loss
