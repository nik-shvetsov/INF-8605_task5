import torch
from model import FoodClassifier, ModelHandler
from dataset import Food11Dataset
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
from torch.utils.data import DataLoader, random_split
from utils import construct_cm


if __name__ == "__main__":
    # Config
    torch.set_float32_matmul_precision(config.MATMUL_PRECISION)

    # Dataloaders
    train_dataset = Food11Dataset(config.TRAIN_DIR, config.ATF, preproc=True, augment=True)
    val_dataset = Food11Dataset(config.VAL_DIR, config.ATF, preproc=True, augment=True)
    test_dataset = Food11Dataset(config.TEST_DIR, config.ATF, preproc=True, augment=True)

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    # Model
    model = FoodClassifier(
        model_name = config.MODEL_NAME,
        pretrained = config.PRETRAINED,
        out_classes = config.NUM_CLASSES
    )

    optimizer = config.OPTIMIZER(model.parameters(), **config.OPTIMIZER_PARAMS)
    scheduler = config.SCHEDULER(optimizer, **config.SCHEDULER_PARAMS)
    
    mh = ModelHandler(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        valid_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=config.ACCELERATOR
    )

    # Train model
    mh.train(accumulate_grad_batches=config.ACCUM_GRAD_BATCHES, save_final=True)

    # Evaluate on test set
    out_eval = mh.evaluate(dataloader=test_dataloader)
    loss = out_eval["avg_loss"]
    accuracy = out_eval["accuracy"]
    precision = out_eval["precision"]
    recall = out_eval["recall"]
    roc_auc = out_eval["roc_auc"]

    print(f"""
    Model evaluation: \t
    Loss = [{loss:0.5f}] \t
    Accuracy = [{(accuracy * 100):0.2f}%] \t
    Precision = [{(precision * 100):0.2f}%] \t
    Recall = [{(recall * 100):0.2f}%] \t
    ROC AUC = [{(roc_auc * 100):0.2f}%] \t
    """)

    ### Confusion matrix for test set
    construct_cm(
        test_dataloader.dataset.targets, 
        out_eval["preds"], 
        test_dataloader.dataset.class_to_idx.keys()
    )
