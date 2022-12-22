import torch
from torch import cat
from models.base_model import DomainDisentangleModel
from experiments.domain_disentangle_losses import *

W1 = 0.8
W2 = 0.8
W3 = 0.001
ALPHA_ENTROPY = 0.1
# weight decay ?


class DomainDisentangleExperiment:  # See point 2. of the project
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        # define one optimizer for each module in the architecture
        self.optimizers = {
            'Gen': torch.optim.Adam(self.model.feature_extractor.parameters(), lr=opt['lr']),
            'Cat_Enc': torch.optim.Adam(self.model.category_encoder.parameters(), lr=opt['lr']),
            'Dom_Enc': torch.optim.Adam(self.model.domain_encoder.parameters(), lr=opt['lr']),
            'Cat_Class': torch.optim.Adam(self.model.category_classifier.parameters(), lr=opt['lr']),
            'Dom_Class': torch.optim.Adam(self.model.domain_classifier.parameters(), lr=opt['lr']),
            'Recon': torch.optim.Adam(self.model.reconstructor.parameters(), lr=opt['lr']),
        }

        # Setup of losses
        self.criterion = [torch.nn.CrossEntropyLoss(), NegHLoss(), torch.nn.CrossEntropyLoss(), NegHLoss(), ReconstructionLoss()]



    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = [
            self.optimizers['Gen'].state_dict(),
            self.optimizers['Cat_Enc'].state_dict(),
            self.optimizers['Dom_Enc'].state_dict(),
            self.optimizers['Cat_Class'].state_dict(),
            self.optimizers['Dom_Class'].state_dict(),
            self.optimizers['Recon'].state_dict(),
        ]

        torch.save(checkpoint, path)


    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizers['Gen'].load_state_dict(checkpoint['optimizer'][0])
        self.optimizers['Cat_Enc'].load_state_dict(checkpoint['optimizer'][1])
        self.optimizers['Dom_Enc'].load_state_dict(checkpoint['optimizer'][2])
        self.optimizers['Cat_Class'].load_state_dict(checkpoint['optimizer'][3])
        self.optimizers['Dom_Class'].load_state_dict(checkpoint['optimizer'][4])
        self.optimizers['Recon'].load_state_dict(checkpoint['optimizer'][5])

        return iteration, best_accuracy, total_train_loss


    def reset_gradient(self):
        for opt in self.optimizers.values():
            opt.zero_grad()

    def optimize_step_on_optimizers(self, optim_key_list):
        for k in optim_key_list:
            self.optimizers[k].step()
        self.reset_gradient()


    def train_iteration(self, data, **loss_acc_logger):
        src_img, category_labels, tgt_img, _ = data
        src_img = src_img.to(self.device)
        category_labels = category_labels.to(self.device)
        tgt_img = tgt_img.to(self.device)

        batch_size = src_img.size(0)

        # Reset all optimizers
        self.reset_gradient()


        # CATEGORY DISENTANGLEMENT
        # Train Category Classifier
        logits = self.model(src_img, 0)
        cat_classif_loss = W1 * self.criterion[0](logits, category_labels)
        cat_classif_loss.backward()
        self.optimize_step_on_optimizers(['Gen', 'Cat_Enc', 'Cat_Class'])

        # Confuse Domain Classifier (freeze DC)
        logits1 = self.model(src_img, 1)
        logits2 = self.model(tgt_img, 1)
        dc_confusion_loss = W1 * self.criterion[1](cat((logits1, logits2), dim=1)) * ALPHA_ENTROPY
        dc_confusion_loss.backward()
        self.optimize_step_on_optimizers(['Gen', 'Cat_Enc'])


        # DOMAIN DISENTANGLEMENT
        # Train Domain Classifier
        logits1 = self.model(src_img, 2)
        # create tensor with scr_domain label = 0
        src_dom_label = torch.full((batch_size,), fill_value=0, device=self.device)

        logits2 = self.model(tgt_img, 2)
        # create tensor with tgt_domain label = 1
        tgt_dom_label = torch.full((batch_size,), fill_value=1, device=self.device)
        dom_classif_loss = W2 * self.criterion[2](cat((logits1, logits2), dim=0), cat((src_dom_label, tgt_dom_label), dim=0))
        dom_classif_loss.backward()
        self.optimize_step_on_optimizers(['Gen', 'Dom_Enc', 'Dom_Class'])

        # Confuse Category Classifier
        logits1 = self.model(src_img, 3)
        logits2 = self.model(tgt_img, 3)
        c_confusion_loss = W2 * self.criterion[3](cat((logits1, logits2), dim=1)) * ALPHA_ENTROPY
        c_confusion_loss.backward()
        self.optimize_step_on_optimizers(['Gen', 'Dom_Enc'])


        # RECONSTRUCTION
        # Extract features from feature extractor
        fG_src = self.model(src_img, -1)
        fG_tgt = self.model(tgt_img, -1)
        logits1 = self.model(src_img, 4)
        loss4a = self.criterion[4](logits1, fG_src)
        logits2 = self.model(tgt_img, 4)
        loss4b = self.criterion[4](logits2, fG_tgt)
        reconstruction_loss = (loss4a + loss4b) / 2 * W3  # is this the correct way??
        reconstruction_loss.backward()
        self.optimize_step_on_optimizers(['Gen', 'Cat_Enc', 'Dom_Enc', 'Recon'])

        loss = cat_classif_loss + dc_confusion_loss + dom_classif_loss + c_confusion_loss + reconstruction_loss

        loss_acc_logger['loss_log']['cat_classif_loss'] += cat_classif_loss
        loss_acc_logger['loss_log']['dc_confusion_entr_loss'] += dc_confusion_loss
        loss_acc_logger['loss_log']['dom_classif_loss'] += dom_classif_loss
        loss_acc_logger['loss_log']['c_confusion_entr_loss'] += c_confusion_loss
        loss_acc_logger['loss_log']['total_loss'] += reconstruction_loss
        loss_acc_logger['train_counter'] += 1

        return loss.item()


    def validate(self, loader, test=False, **loss_acc_logger):
        self.model.eval()
        category_accuracy = 0
        category_count = 0
        domain_accuracy = 0
        domain_count = 0
        loss = 0
        with torch.no_grad():
            for src_img, category_labels, tgt_img, tgt_category_labels in loader:
                src_img = src_img.to(self.device)
                category_labels = category_labels.to(self.device)
                tgt_img = tgt_img.to(self.device)
                tgt_category_labels = tgt_category_labels.to(self.device)

                batch_size = src_img.size(0)

                # Extract features from feature extractor
                fG_src = self.model(src_img, -1)
                fG_tgt = self.model(tgt_img, -1)

                # CATEGORY CLASSIFICATION (only on src_img)
                logits = self.model(src_img, 0)
                cat_classif_loss1 = W1 * self.criterion[0](logits, category_labels)
                pred = torch.argmax(logits, dim=-1)
                category_accuracy += (pred == category_labels).sum().item()
                category_count += src_img.size(0)

                # If testing check category label on target source too
                if test:
                    logits = self.model(tgt_img, 0)
                    cat_classif_loss2 = W1 * self.criterion[0](logits, tgt_category_labels)
                    pred = torch.argmax(logits, dim=-1)
                    category_accuracy += (pred == category_labels).sum().item()
                    category_count += src_img.size(0)
                else:
                    cat_classif_loss2 = 0

                cat_classif_loss = (cat_classif_loss1 + cat_classif_loss2) / 2

                # Confuse Domain Classifier
                logits1 = self.model(src_img, 1)
                logits2 = self.model(tgt_img, 1)
                dc_confusion_loss = W1 * self.criterion[1](cat((logits1, logits2), dim=1)) * ALPHA_ENTROPY

                # DOMAIN CLASSIFICATION
                logits1 = self.model(src_img, 2)
                # create tensor with scr_domain label = 0
                src_dom_label = torch.full((batch_size,), fill_value=0, device=self.device)
                pred = torch.argmax(logits1, dim=-1)
                domain_accuracy += (pred == src_dom_label).sum().item()
                domain_count += src_img.size(0)

                logits2 = self.model(tgt_img, 2)
                # create tensor with tgt_domain label = 1
                tgt_dom_label = torch.full((batch_size,), fill_value=1, device=self.device)
                pred = torch.argmax(logits2, dim=-1)
                domain_accuracy += (pred == tgt_dom_label).sum().item()
                domain_count += src_img.size(0)
                dom_classif_loss = W2 * self.criterion[2](cat((logits1, logits2), dim=0), cat((src_dom_label, tgt_dom_label), dim=0))

                # Confuse Category Classifier
                logits1 = self.model(src_img, 3)
                logits2 = self.model(tgt_img, 3)
                c_confusion_loss = W2 * self.criterion[3](cat((logits1, logits2), dim=1)) * ALPHA_ENTROPY

                # RECONSTRUCTION
                logits1 = self.model(src_img, 4)
                loss4a = self.criterion[4](logits1, fG_src)
                logits2 = self.model(tgt_img, 4)
                loss4b = self.criterion[4](logits2, fG_tgt)
                reconstruction_loss = (loss4a + loss4b) / 2 * W3  # is this the correct way??

                loss += cat_classif_loss + dc_confusion_loss + dom_classif_loss + c_confusion_loss + reconstruction_loss



        mean_accuracy = category_accuracy / category_count
        mean_loss = loss / category_count
        mean_domain_acc = domain_accuracy / domain_count

        loss_acc_logger['loss_log_val']['cat_classif_loss'] += cat_classif_loss
        loss_acc_logger['loss_log_val']['dc_confusion_entr_loss'] += dc_confusion_loss
        loss_acc_logger['loss_log_val']['dom_classif_loss'] += dom_classif_loss
        loss_acc_logger['loss_log_val']['c_confusion_entr_loss'] += c_confusion_loss
        loss_acc_logger['loss_log_val']['total_loss'] += mean_loss

        loss_acc_logger['acc_logger']['cat_classif_acc'] += mean_accuracy
        loss_acc_logger['acc_logger']['dom_classif_acc'] += mean_domain_acc
        loss_acc_logger['val_counter'] += 1

        self.model.train()
        return mean_accuracy, mean_loss
