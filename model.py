import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import timm
from pprint import pprint
from tqdm import tqdm, trange
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_auroc, multiclass_precision, multiclass_recall
import config
from utils import test_net, save_run_config
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import lovely_tensors as lt

lt.monkey_patch()
lt.set_config(sci_mode=False)
torch.set_printoptions(sci_mode=False)


class FoodClassifier(nn.Module):
    def __init__(self, model_name, pretrained, out_classes):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_classes, features_only=False)

    def forward(self, image):
        output = self.model(image)
        return output

class ModelHandler():
    def __init__(self, model, loss_fn=None, optimizer=None, scheduler=None, train_dataloader=None, 
                    valid_dataloader=None, test_dataloader=None, device=None):
        super().__init__()
        self.device = device if device is not None else 'cpu'
        self.model = model.to(device=self.device)

        self.criterion = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler()

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        if config.EARLY_STOP:
            self.patience = config.PATIENCE
            self.best_trained_model = {
                "score": 0,
                "state_dict": None,
                "current_pateince": self.patience,
            }

    def test_step(self, imgs, targets):
        self.model.eval()
        logits = self.model(imgs)

        (pred_logits, pred_idxs) = logits.max(1)

        return {
            "logits": logits,
            "preds": pred_idxs
        }

    def eval_step(self, imgs, targets):
        self.model.eval()

        if config.USE_AMP:
            with torch.amp.autocast(device_type=config.ACCELERATOR, cache_enabled=True):
                logits = self.model(imgs)
                loss = self.criterion(logits, targets)
        else:
            logits = self.model(imgs)
            loss = self.criterion(logits, targets)

        (pred_logits, pred_idxs) = logits.max(1)

        return {
            "loss": float(loss.item()),
            "preds": pred_idxs,
            "class_probs": F.softmax(logits, dim=1)
        }

    def evaluate(self, dataloader):
        total_loss = 0
        preds = []
        class_probs = []

        with torch.no_grad():
            for batch_idx, (imgs, targets) in enumerate(dataloader):
                imgs = imgs.type(torch.FloatTensor).to(device=self.device)
                targets = targets.type(torch.LongTensor).to(device=self.device)

                out_eval_step = self.eval_step(imgs, targets)
                
                batch_loss = out_eval_step["loss"]
                total_loss += batch_loss

                preds.extend(out_eval_step["preds"].tolist())

                class_probs.append(out_eval_step["class_probs"])
                    
        accuracy = multiclass_accuracy(torch.tensor(preds), torch.tensor(dataloader.dataset.targets), num_classes=config.NUM_CLASSES, average='micro')
        precision = multiclass_precision(torch.tensor(preds), torch.tensor(dataloader.dataset.targets), num_classes=config.NUM_CLASSES, average='macro')
        recall = multiclass_recall(torch.tensor(preds), torch.tensor(dataloader.dataset.targets), num_classes=config.NUM_CLASSES, average='macro')
        roc_auc = multiclass_auroc(torch.tensor(torch.cat(class_probs, dim=0).numpy(force=True)), torch.tensor(dataloader.dataset.targets), num_classes=config.NUM_CLASSES, average='macro')
        
        return {
            'avg_loss': total_loss / len(dataloader),
            'accuracy': float(accuracy.item()),
            'precision': float(precision.item()),
            'recall': float(recall.item()),
            'roc_auc': float(roc_auc.item()),
            'preds': preds,
        }

    def train_step(self, imgs, targets, grad_step=True, accumulate_norm_factor=1):
        self.model.train()

        # Forward
        if config.USE_AMP:
            with torch.amp.autocast(device_type=config.ACCELERATOR, cache_enabled=True):
                logits = self.model(imgs)
                loss = self.criterion(logits, targets)
        else:
            logits = self.model(imgs)
            loss = self.criterion(logits, targets)

        if config.USE_GRAD_SCALER:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if grad_step: self.optimizer.step()
                
        if grad_step: self.optimizer.zero_grad()
        return float(loss.item()) / accumulate_norm_factor
    
    def train(self, train_dataloader=None, valid_dataloader=None, accumulate_grad_batches=1, save_final=False):
        if train_dataloader is None: train_dataloader = self.train_dataloader
        if valid_dataloader is None: valid_dataloader = self.valid_dataloader

        self.model = self.model.to(self.device)
        for epoch in trange(1, config.NUM_EPOCHS + 1, desc='Epochs'):
            train_loss = 0 # train epoch loss
            for batch_idx, (imgs, targets) in enumerate(tqdm(train_dataloader, desc='Batches', leave=False)):

                imgs = imgs.type(torch.FloatTensor).to(device=self.device)
                targets = targets.type(torch.LongTensor).to(device=self.device)
                do_grad_step = ((batch_idx + 1) % accumulate_grad_batches == 0) or (batch_idx + 1 == len(train_dataloader))

                # train batch loss
                train_loss += self.train_step(imgs, targets, grad_step=do_grad_step, accumulate_norm_factor=accumulate_grad_batches) 

            out_eval = self.evaluate(valid_dataloader)

            loss = out_eval["avg_loss"]
            accuracy = out_eval["accuracy"]
            precision = out_eval["precision"]
            recall = out_eval["recall"]
            roc_auc = out_eval["roc_auc"]
   
            tqdm.write(f"""
            Epoch [{epoch}]: \t 
            train loss = [{train_loss:0.5f}] \t 
            val loss = [{loss:0.5f}] \t 
            val accuracy = [{(accuracy * 100):0.2f}%] \t
            val precision = [{(precision * 100):0.2f}%] \t
            val recall = [{(recall * 100):0.2f}%] \t
            val ROC AUC = [{(roc_auc * 100):0.2f}%] \t
            """) 
            
            # Scheduler step
            if self.scheduler is not None: 
                if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.step(loss)
                elif type(self.scheduler) == torch.optim.lr_scheduler.MultiStepLR:
                    self.scheduler.step()
            
            # Early stopping
            if self.best_trained_model is not None:
                if accuracy > self.best_trained_model['score']:
                    self.best_trained_model['score'] = accuracy
                    self.best_trained_model['model_state_dict'] = self.model.state_dict().copy()
                    self.best_trained_model['current_pateince'] = self.patience
                else:
                    self.best_trained_model['current_pateince'] -= 1
                if self.best_trained_model['current_pateince'] < 0:
                    tqdm.write(f"Early stopping at epoch {epoch}")
                    break
                
        if save_final:
            if self.best_trained_model is not None:
                self.model.load_state_dict(self.best_trained_model['model_state_dict'])
            
            out_eval = self.evaluate(self.test_dataloader if self.test_dataloader is not None else self.valid_dataloader)
            accuracy = out_eval["accuracy"]
            
            print(f"Saving model {config.CONFIG_ID}: \"{config.MODEL_NAME}\" initialized with \"{'pretrained' if config.PRETRAINED else 'random'}\" with test set accuracy score = {(accuracy * 100):0.2f}")
            self.save(f'{config.MODEL_FOLDER}/{config.CONFIG_ID}_{(accuracy * 100):0.2f}.pt')
        
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_run_config(path)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def predict(self, pil_img, show_results=False, label_dict=None, fname=None):
        self.model = self.model.to(self.device)
        self.model.eval()

        preproc_img = {'image': pil_img}
        preproc_img['image'] = config.ATF['preproc'](preproc_img['image'])
        preproc_img['image'] = np.asarray(preproc_img['image'].permute(1, 2, 0))
        preproc_img = config.ATF['resize_to_tensor'](image=preproc_img['image'])['image']
        img = preproc_img.unsqueeze(0).type(torch.FloatTensor).to(device=self.device)
        with torch.no_grad():            
            logits = self.model(img)
            probs = F.softmax(logits, dim=1)
            (prob, prediction) = probs.max(1)
            # (_, prediction) = logits.max(1)
            
            prob = float(prob)
            prediction = int(prediction)
            
            if show_results:
                plt.figure(figsize=(10, 5))
                plt.title(prediction) if label_dict is None else plt.title(f"{label_dict[prediction]} ({prediction}): {prob:.2f}")
                plt.imshow(pil_img)
                plt.show()
                if fname is not None: plt.savefig(fname)
            
            else:
                return prediction, prob


if __name__ == '__main__':
    model = FoodClassifier(
        model_name = config.MODEL_NAME,
        pretrained = config.PRETRAINED,
        out_classes = config.NUM_CLASSES
    ).eval()

    pprint(model.state_dict)
    test_net(model, device='cpu', size=(3,256,256), n_batch=4)
    
    
