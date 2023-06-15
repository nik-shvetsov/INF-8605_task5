import torch

CONFIGS = {
    'resnet18_1': {
        "MODEL_NAME": 'resnet18',
        "PRETRAINED": True,
        "NUM_CLASSES": 11,

        "NUM_EPOCHS": 50,
        "BATCH_SIZE": 64,
        "ACCUM_GRAD_BATCHES": 1,

        "USE_AMP": True,
        "USE_GRAD_SCALER": True,
        "EARLY_STOP": True,
        "PATIENCE": 10,
        "OPTIMIZER": torch.optim.Adam,
        "OPTIMIZER_PARAMS": {
            "lr": 0.001,
            "weight_decay": 5e-4
        },
        "SCHEDULER": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "SCHEDULER_PARAMS": {
            "mode": 'min', 
            "factor": 0.1,
            "patience": 5, 
            "verbose": True,
        },

        "INPUT_SIZE": (256, 256),
        "NUM_WORKERS": 8,

        "TRAIN_DIR": "/code/INF-8605/lab_assignment/food11_multiclass/split/training",
        "VAL_DIR": "/code/INF-8605/lab_assignment/food11_multiclass/split/validation",
        "TEST_DIR": "/code/INF-8605/lab_assignment/food11_multiclass/split/evaluation",

        "ACCELERATOR": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "MATMUL_PRECISION": "medium"
    },
}
