import torch
import datasets
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from torch.optim import AdamW
from torch.nn import functional as F
from uniformer import uniformer_small, uniformer_base
from huggingface_hub import hf_hub_download
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_loss_history(train_loss_history, dev_loss_history=None):
    plt.plot(train_loss_history)
    if dev_loss_history is not None: plt.plot(dev_loss_history)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    legend = ['Train loss']
    if dev_loss_history is not None: legend.append('Dev loss')
    plt.legend(legend)
    plt.savefig('loss.pdf', dpi=300, bbox_inches='tight')

def generate_confusion_matrix(config, predictions, references):
    cm_path = config["cm_save_path"]
    class_names = config["class_names"]
    
    cm = confusion_matrix(references, predictions, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot()
    plt.savefig(cm_path)

class VideoTransformer(torch.nn.Module):
    def __init__(self, config):
        super(VideoTransformer, self).__init__()
        learning_rate = config["learning_rate"]
        self.use_cuda = config["use_cuda"]
        weight_decay = config["weight_decay"]
        binarise = config["binarise"]
        class_weights = None
        if binarise:
            class_weights = torch.as_tensor(config["class_weights"]).float()
            if self.use_cuda:
                class_weights = class_weights.cuda()

        self.model = uniformer_base()
        # load state
        #model_path = hf_hub_download(repo_id="Sense-X/uniformer_video", filename="uniformer_small_k400_16x8.pth")
        model_path = hf_hub_download(repo_id="Sense-X/uniformer_video", filename="uniformer_base_k400_16x8.pth")
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.embed_dim = 512
        self.model.reset_classifier(num_classes=2 if binarise else 11)

        # set to eval mode
        self.model = self.model.eval()
        if self.use_cuda:
            self.model = self.model.cuda()

        self.loss_fct = torch.nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=-1
        )
        self.optimiser = AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.metrics = {
            'f1': datasets.load_metric('f1'),
            'accuracy': datasets.load_metric('accuracy')
        }
        if binarise:
            self.metrics['recall'] = datasets.load_metric('recall')
            self.metrics['precision'] = datasets.load_metric('precision')

    def forward(self, x):
        return self.model(x)
    
    def feedforward_pass(self, x, labels):
        if self.use_cuda:
            x = x.cuda()
            labels = labels.cuda()
        logits = self.forward(x)
        return logits.detach().cpu().numpy(), self.loss_fct(logits, labels)

    def training_step(self, x, labels):
        self.optimiser.zero_grad()
        _, loss = self.feedforward_pass(x, labels)
        loss.backward()
        self.optimiser.step()
        return loss.item()

    def train_1_epoch(self, dataset):
        self.model.train()
        loss_history = []
        n_samples = len(dataset.dataloader)
        for batch in tqdm(dataset.dataloader, total=n_samples):
            loss = self.training_step(batch["data"], batch["label"])
            loss_history.append(loss)
        print('TRAIN Average loss: {:.2f}'.format(np.mean(loss_history)))
        return loss_history

    def evaluate(self, dataset):
        self.model.eval()
        loss_history, predictions, references = [], [], []
        n_samples = len(dataset.dataloader)
        for batch in tqdm(dataset.dataloader, total=n_samples):
            with torch.no_grad():
                logits, loss = self.feedforward_pass(batch["data"], batch["label"])
            loss_history.append(loss.item())
            predictions.extend(logits.argmax(-1))
            references.extend(batch["label"].numpy())
        print('DEV Average loss: {:.2f}'.format(np.mean(loss_history)))
        predictions = np.asarray(predictions)
        references = np.asarray(references)
        scores = {}
        for name, metric in self.metrics.items():
            if name == 'f1':
                scores[name] = list(metric.compute(
                predictions=predictions, references=references,
                average='macro'
            ).values())[0]
            else:
                scores[name] = list(metric.compute(
                    predictions=predictions, references=references
                ).values())[0]
            print('{}: {:.2f}'.format(
                name, 
                scores[name]*100
            ))
        return scores

    def train(self, config, train_dataloader, dev_dataloader=None):
        epochs = config["epochs"]
        patience = config["patience"]
        if dev_dataloader is not None:
            self.evaluate(dev_dataloader)
        best_f1_score = 0.
        loss_history = []
        for e in range(epochs):
            epoch_loss_history = self.train_1_epoch(train_dataloader)
            loss_history.extend(epoch_loss_history)
            if dev_dataloader is not None:
                scores = self.evaluate(dev_dataloader)
                if scores["f1"] > best_f1_score:
                    best_f1_score = scores["f1"]
                    patience = config["patience"]
                else:
                    if patience == 0:
                        print("F1 did not improve. Training stopped.")
                        break
                    else:
                        patience = patience - 1
        save_loss_history(loss_history)
                

    def test(self, config, dataset):
        self.model.eval()
        self.batch_size_test = config["batch_size_test"]
        TP, FP, TN, FN = 0, 0, 0, 0
        nb_samples = 0
        for batch in tqdm(dataset.dataloader):
            # For each video
            for i in range(self.batch_size_test):
                nb_samples += 1
                seq_len = batch["label"].shape[0]
                fall_appear, fall_predicted, seq_finished = False, False, False
                for j in range(seq_len):
                    if int(batch["label"][j,i]) == -1: 
                        seq_finished = True
                        break
                    with torch.no_grad():
                        logits, _ = self.feedforward_pass(
                            batch["data"][j,i,...].unsqueeze(0),
                            batch["label"][j,i].unsqueeze(0)
                        )
                    pred = int(logits.argmax(-1)[0])
                    # Fall has occurred
                    if int(batch["label"][j,i]) == 1:
                        fall_appear = True
                        # Fall predicted
                        if pred == 1:
                            TP += 1
                            fall_predicted = True
                            break
                    # A fall has incorrectly been predicted
                    elif int(batch["label"][j,i]) == 0 and pred == 1:
                        FP += 1
                        break
                    
                    if j == seq_len-1:
                        # A fall appeared but it was not predicted
                        if fall_appear and not fall_predicted:
                            FN += 1
                        # A fall did not appear and it was not predicted
                        elif not fall_appear and not fall_predicted:
                            TN += 1
                        break
                if seq_finished:
                    # A fall appeared but it was not predicted
                    if fall_appear and not fall_predicted:
                        FN += 1
                    # A fall did not appear and it was not predicted
                    elif not fall_appear and not fall_predicted:
                        TN += 1
        
        accuracy = (TP+TN) / (TP+TN+FP+FN)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        specificity = TN / (TN+FP)
        denom = precision+recall
        f1 = 0 if denom == 0 else 2* ((precision*recall) / denom)
        print('TEST ------------------')
        print('Accuracy: {:.2f}'.format(accuracy*100))
        print('Recall/Sensitivity: {:.2f}'.format(recall*100))
        print('Precision: {:.2f}'.format(precision*100))
        print('Specificity {:.2f}'.format(specificity*100))
        print('F1 score: {:.2f}'.format(f1*100))

    def test_by_clip(self, config, dataset):
        self.model.eval()
        loss_history, predictions, references = [], [], []
        TP, TN, FN, FP = 0, 0, 0, 0
        n_samples = len(dataset.dataloader)
        for batch in tqdm(dataset.dataloader, total=n_samples):
            seq_len = batch["label"].shape[0]
            for i in range(seq_len):
                with torch.no_grad():
                    logits, loss = self.feedforward_pass(batch["data"][i,:], batch["label"][i,:]) 
                loss_history.append(loss.item())
                class_preds = logits.argmax(-1)
                for j in range(len(batch["label"][i,:])):
                    if int(batch["label"][i,j]) != -1:
                        predictions.append(class_preds[j])
                        references.append(int(batch["label"][i,j].numpy()))
                batch_TP, batch_TN, batch_FN, batch_FP = compute_metrics(
                    class_preds, batch["label"][i,:], config
                )
                TP += batch_TP
                TN += batch_TN
                FN += batch_FN
                FP += batch_FP
        print('TEST Average loss (by clip): {:.2f}'.format(np.mean(loss_history)))
        generate_confusion_matrix(config, predictions, references)
        predictions = np.asarray(predictions)
        references = np.asarray(references)
        scores = {}
        for name, metric in self.metrics.items():
            if name == 'f1':
                scores[name] = list(metric.compute(
                    predictions=predictions, references=references,
                    average='macro'
                ).values())[0]
            else:
                scores[name] = list(metric.compute(
                    predictions=predictions, references=references
                ).values())[0]
            print('{}: {:.2f}'.format(
                name, 
                scores[name]*100
            ))
      
        return scores
