import torch
import clip
from torch import cat
from models.base_model import ClipDisentangleModel, ClipDisentangleModel_DomainGeneralization
from experiments.domain_disentangle_losses import *

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


W1 = 0.99
W2 = 0.099  # Being used for all "domain" related losses (DomEnc, DomClassif, DomEntropy, and Clip)
W3 = 0.001
ALPHA_ENTROPY = 0.7

class CLIPDisentangleExperiment: # See point 4. of the project
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = ClipDisentangleModel() if not opt['domain_generalization'] else ClipDisentangleModel_DomainGeneralization()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Load CLIP model and freeze it
        self.clip_model, _ = clip.load('ViT-B/32', device='cpu')  # load it first to CPU to ensure you're using fp32 precision.
        self.clip_model.to(self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Warmup counter
        self.warmup_counter = 0

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
        self.criterion = [torch.nn.CrossEntropyLoss(), NegHLoss(), torch.nn.CrossEntropyLoss(),
                          NegHLoss(), ReconstructionLoss(), torch.nn.MSELoss()]



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
        if self.opt['domain_generalization']:
            img, category_labels, domain_labels, tokenized_text = data
            img = img.to(self.device)
            category_labels = category_labels.to(self.device)
            domain_labels = domain_labels.to(self.device)
            tokenized_text = tokenized_text.to(self.device)

            # Reset all optimizers
            self.reset_gradient()

            # CATEGORY DISENTANGLEMENT
            # Train Category Classifier
            logits = self.model(img, 0)
            cat_classif_loss = W1 * self.criterion[0](logits, category_labels)
            cat_classif_loss.backward()
            self.optimize_step_on_optimizers(['Gen', 'Cat_Enc', 'Cat_Class'])

            # Confuse Domain Classifier (freeze DC)
            logits = self.model(img, 1)
            dc_confusion_loss = W1 * self.criterion[1](logits) * ALPHA_ENTROPY
            dc_confusion_loss.backward()
            self.optimize_step_on_optimizers(['Gen', 'Cat_Enc'])

            # CLIP DISENTANGLEMENT
            dom_features = self.model(img, 4)
            text_features = self.clip_model.encode_text(tokenized_text)
            lossClip = self.criterion[5](dom_features, text_features) * W2
            lossClip.backward()
            self.optimize_step_on_optimizers(['Gen', 'Dom_Enc'])

            # DOMAIN DISENTANGLEMENT
            # Train Domain Classifier
            logits = self.model(img, 2)
            dom_classif_loss = W2 * self.criterion[2](logits, domain_labels)
            dom_classif_loss.backward()
            self.optimize_step_on_optimizers(['Gen', 'Dom_Enc', 'Dom_Class'])

            # Confuse Category Classifier
            logits = self.model(img, 3)
            c_confusion_loss = W2 * self.criterion[3](logits) * ALPHA_ENTROPY
            c_confusion_loss.backward()
            self.optimize_step_on_optimizers(['Gen', 'Dom_Enc'])

            # RECONSTRUCTION
            # Extract features from feature extractor
            fG_src = self.model(img, -1)
            logits = self.model(img, 4)
            loss4 = self.criterion[4](logits, fG_src)
            reconstruction_loss = loss4 * W3
            reconstruction_loss.backward()
            self.optimize_step_on_optimizers(['Gen', 'Cat_Enc', 'Dom_Enc', 'Recon'])

        else:
            src_img, category_labels, tokenized_text_src, tgt_img, _, tokenized_text_tgt = data
            src_img = src_img.to(self.device)
            category_labels = category_labels.to(self.device)
            tgt_img = tgt_img.to(self.device)
            tokenized_text_src = tokenized_text_src.to(self.device)
            tokenized_text_tgt = tokenized_text_tgt.to(self.device)

            batch_size = src_img.size(0)

            # Reset all optimizers
            self.reset_gradient()

            # WARMUP: Train the category classifier and the domain classifier alternatively
            if self.warmup_counter < 1200:
                self.warmup_counter += 1

                # Train Category Classifier
                logits = self.model(src_img, 0)
                cat_classif_loss = self.criterion[0](logits, category_labels)
                cat_classif_loss.backward()
                self.optimize_step_on_optimizers(['Cat_Enc', 'Cat_Class'])

                # Train Domain Classifier
                logits1 = self.model(src_img, 2)
                # create tensor with scr_domain label = 0
                src_dom_label = torch.full((batch_size,), fill_value=0, device=self.device)
                logits2 = self.model(tgt_img, 2)
                # create tensor with tgt_domain label = 1
                tgt_dom_label = torch.full((batch_size,), fill_value=1, device=self.device)
                dom_classif_loss = self.criterion[2](cat((logits1, logits2), dim=0),
                                                     cat((src_dom_label, tgt_dom_label), dim=0))
                dom_classif_loss.backward()
                self.optimize_step_on_optimizers(['Dom_Enc', 'Dom_Class'])
                return cat_classif_loss.item() + dom_classif_loss.item()

            if self.warmup_counter == 1200:
                print("Finished warmup")

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
            self.optimize_step_on_optimizers(['Cat_Enc'])


            # CLIP DISENTANGLEMENT
            dom_features_src = self.model(src_img, 4)
            text_features_src = self.clip_model.encode_text(tokenized_text_src)
            lossClip1 = self.criterion[5](dom_features_src, text_features_src)

            dom_features_tgt = self.model(tgt_img, 4)
            text_features_tgt = self.clip_model.encode_text(tokenized_text_tgt)
            lossClip2 = self.criterion[5](dom_features_tgt, text_features_tgt)

            lossClip = (lossClip1 + lossClip2) * W2
            lossClip.backward()
            self.optimize_step_on_optimizers(['Gen', 'Dom_Enc'])


            # DOMAIN DISENTANGLEMENT
            # Train Domain Classifier
            logits1 = self.model(src_img, 2)
            # create tensor with scr_domain label = 0
            src_dom_label = torch.full((batch_size,), fill_value=0, device=self.device)

            logits2 = self.model(tgt_img, 2)
            # create tensor with tgt_domain label = 1
            tgt_dom_label = torch.full((batch_size,), fill_value=1, device=self.device)

            dom_classif_loss = W2 * self.criterion[2](cat((logits1, logits2), dim=0),
                                                                  cat((src_dom_label, tgt_dom_label), dim=0))
            dom_classif_loss.backward()
            self.optimize_step_on_optimizers(['Gen', 'Dom_Enc', 'Dom_Class'])

            # Confuse Category Classifier
            logits1 = self.model(src_img, 3)
            logits2 = self.model(tgt_img, 3)
            c_confusion_loss = W2 * self.criterion[3](cat((logits1, logits2), dim=1)) * ALPHA_ENTROPY
            c_confusion_loss.backward()
            self.optimize_step_on_optimizers(['Dom_Enc'])


            # RECONSTRUCTION
            # Extract features from feature extractor
            fG_src = self.model(src_img, -1)
            fG_tgt = self.model(tgt_img, -1)
            logits1 = self.model(src_img, 4)
            loss4a = self.criterion[4](logits1, fG_src)
            logits2 = self.model(tgt_img, 4)
            loss4b = self.criterion[4](logits2, fG_tgt)
            reconstruction_loss = (loss4a + loss4b) / 2 * W3
            reconstruction_loss.backward()
            self.optimize_step_on_optimizers(['Cat_Enc', 'Dom_Enc', 'Recon'])

        loss = cat_classif_loss + dc_confusion_loss + dom_classif_loss + c_confusion_loss + reconstruction_loss + lossClip

        loss_acc_logger['loss_log']['cat_classif_loss'] += cat_classif_loss
        loss_acc_logger['loss_log']['dc_confusion_entr_loss'] += dc_confusion_loss
        loss_acc_logger['loss_log']['dom_classif_loss'] += dom_classif_loss
        loss_acc_logger['loss_log']['c_confusion_entr_loss'] += c_confusion_loss
        loss_acc_logger['loss_log']['reconstr_loss'] += reconstruction_loss
        loss_acc_logger['loss_log']['total_loss'] += loss
        loss_acc_logger['train_counter'] += 1

        self.warmup_counter += 1
        return loss.item()


    def validate(self, loader, test=False, **loss_acc_logger):
        self.model.eval()
        category_accuracy = 0
        category_count = 0
        domain_accuracy = 0
        domain_count = 0
        loss = 0
        with torch.no_grad():
            if self.opt['domain_generalization']:
                 for img, category_labels, domain_labels, tokenized_text in loader:
                    img = img.to(self.device)
                    category_labels = category_labels.to(self.device)
                    domain_labels = domain_labels.to(self.device)
                    tokenized_text = tokenized_text.to(self.device)

                    batch_size = img.size(0)

                    # Extract features from feature extractor
                    fG = self.model(img, -1)

                    # CATEGORY CLASSIFICATION (only on src_img)
                    logits = self.model(img, 0)
                    cat_classif_loss = W1 * self.criterion[0](logits, category_labels)
                    pred = torch.argmax(logits, dim=-1)
                    category_accuracy += (pred == category_labels).sum().item()
                    category_count += img.size(0)

                    # Confuse Domain Classifier
                    logits = self.model(img, 1)
                    dc_confusion_loss = W1 * self.criterion[1](logits) * ALPHA_ENTROPY

                    # DOMAIN CLASSIFICATION
                    logits = self.model(img, 2)
                    pred = torch.argmax(logits, dim=-1)
                    domain_accuracy += (pred == domain_labels).sum().item()
                    domain_count += img.size(0)
                    dom_classif_loss = W2 * self.criterion[2](logits, domain_labels)

                    # Confuse Category Classifier
                    logits = self.model(img, 3)
                    c_confusion_loss = W2 * self.criterion[3](logits) * ALPHA_ENTROPY

                    # RECONSTRUCTION
                    logits = self.model(img, 4)
                    loss4 = self.criterion[4](logits, fG)
                    reconstruction_loss = loss4 * W3

                    # CLIP DISENTANGLEMENT
                    dom_features = self.model(img, 4)
                    text_features = self.clip_model.encode_text(tokenized_text)
                    lossClip = self.criterion[5](dom_features, text_features) * W2

                    loss += cat_classif_loss + dc_confusion_loss + dom_classif_loss + c_confusion_loss + reconstruction_loss + lossClip

            else:
                for src_img, category_labels, tokenized_text_src, tgt_img, tgt_category_labels, tokenized_text_tgt in loader:

                    src_img = src_img.to(self.device)
                    category_labels = category_labels.to(self.device)
                    tgt_img = tgt_img.to(self.device)
                    tgt_category_labels = tgt_category_labels.to(self.device)
                    tokenized_text_src = tokenized_text_src.to(self.device)
                    tokenized_text_tgt = tokenized_text_tgt.to(self.device)

                    batch_size = src_img.size(0)

                    # Extract features from feature extractor
                    fG_src = self.model(src_img, -1)
                    fG_tgt = self.model(tgt_img, -1)

                    # CATEGORY CLASSIFICATION (only on src_img during validation)
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
                        category_accuracy += (pred == tgt_category_labels).sum().item()
                        category_count += tgt_img.size(0)
                        cat_classif_loss = (cat_classif_loss1 + cat_classif_loss2) / 2
                    else:
                        cat_classif_loss = cat_classif_loss1


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
                    domain_count += tgt_img.size(0)
                    dom_classif_loss = W2 * self.criterion[2](cat((logits1, logits2), dim=0),
                                                                          cat((src_dom_label, tgt_dom_label), dim=0))


                    # Confuse Category Classifier
                    logits1 = self.model(src_img, 3)
                    logits2 = self.model(tgt_img, 3)
                    c_confusion_loss = W2 * self.criterion[3](cat((logits1, logits2), dim=1)) * ALPHA_ENTROPY

                    # RECONSTRUCTION
                    logits1 = self.model(src_img, 4)
                    loss4a = self.criterion[4](logits1, fG_src)
                    logits2 = self.model(tgt_img, 4)
                    loss4b = self.criterion[4](logits2, fG_tgt)
                    reconstruction_loss = (loss4a + loss4b) / 2 * W3

                    # Clip disentanglement
                    dom_features_src = self.model(src_img, 4)
                    text_features_src = self.clip_model.encode_text(tokenized_text_src)
                    lossClip1 = self.criterion[5](dom_features_src, text_features_src)

                    dom_features_tgt = self.model(tgt_img, 4)
                    text_features_tgt = self.clip_model.encode_text(tokenized_text_tgt)
                    lossClip2 = self.criterion[5](dom_features_tgt, text_features_tgt)


                    lossClip = (lossClip1 + lossClip2) * W2

                    loss += cat_classif_loss + dc_confusion_loss + dom_classif_loss + c_confusion_loss + reconstruction_loss + lossClip



        mean_accuracy = category_accuracy / category_count
        mean_loss = loss / category_count
        mean_domain_acc = domain_accuracy / domain_count

        loss_acc_logger['loss_log_val']['cat_classif_loss'] += cat_classif_loss
        loss_acc_logger['loss_log_val']['dc_confusion_entr_loss'] += dc_confusion_loss
        loss_acc_logger['loss_log_val']['dom_classif_loss'] += dom_classif_loss
        loss_acc_logger['loss_log_val']['c_confusion_entr_loss'] += c_confusion_loss
        loss_acc_logger['loss_log_val']['reconstr_loss'] += reconstruction_loss
        loss_acc_logger['loss_log_val']['total_loss'] += mean_loss

        loss_acc_logger['acc_logger']['cat_classif_acc'] += mean_accuracy
        loss_acc_logger['acc_logger']['dom_classif_acc'] += mean_domain_acc
        loss_acc_logger['val_counter'] += 1

        self.model.train()
        return mean_accuracy, mean_loss


    def test_on_target(self, loader):
        print("testing on tgt only")
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x, 0)
                loss += self.criterion[0](logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        print("acc = ", mean_accuracy)
        return mean_accuracy, mean_loss


    def tSNE_plot(self, loader, extract_features_branch=0, iter=0, base_path=''):
        dataset = []
        labels = []
        for data in loader:
            src_img, _, _, tgt_img, _, _ = data
            src_img = src_img.to(self.device)
            tgt_img = tgt_img.to(self.device)
            fG_src = self.model(src_img, extract_features_branch)  # from category encoder
            fG_tgt = self.model(tgt_img, extract_features_branch)
            fG_src = fG_src.detach().cpu().numpy()
            fG_tgt = fG_tgt.detach().cpu().numpy()
            for el in fG_src:
                dataset.append(el)
                labels.append(0)
            for el in fG_tgt:
                dataset.append(el)
                labels.append(1)
        dataset = np.asarray(dataset)
        tsne = TSNE()  # t-Distributed Stochastic Neighbor Embedding
        tsne_results = tsne.fit_transform(dataset)
        src_x_coords = []
        src_y_coords = []
        tgt_x_coords = []
        tgt_y_coords = []
        i = 0
        for l in labels:
            if l == 0:
                src_x_coords.append(tsne_results[i][0])
                src_y_coords.append(tsne_results[i][1])
            else:
                tgt_x_coords.append(tsne_results[i][0])
                tgt_y_coords.append(tsne_results[i][1])
            i += 1
        src = plt.scatter(src_x_coords, src_y_coords, c='blue', alpha=0.5, s=10)
        tgt = plt.scatter(tgt_x_coords, tgt_y_coords, c='red', alpha=0.5, s=10)
        plt.legend((src, tgt),
                   ('source', 'target'),
                   scatterpoints=1,
                   loc='upper right',
                   ncol=1,
                   fontsize=6)
        # plt.show()
        plt.savefig(f'{base_path}_tSNE_at_iter_{iter}.png')
        plt.clf()
