from pathlib import Path
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from configs import CONFIGS


CONFIG_ID = 'resnet18_1'

MODEL_NAME = CONFIGS[CONFIG_ID]["MODEL_NAME"]
PRETRAINED = CONFIGS[CONFIG_ID]["PRETRAINED"]
NUM_CLASSES = CONFIGS[CONFIG_ID]["NUM_CLASSES"]
NUM_EPOCHS = CONFIGS[CONFIG_ID]["NUM_EPOCHS"]
BATCH_SIZE = CONFIGS[CONFIG_ID]["BATCH_SIZE"]
ACCUM_GRAD_BATCHES = CONFIGS[CONFIG_ID]["ACCUM_GRAD_BATCHES"]
USE_AMP = CONFIGS[CONFIG_ID]["USE_AMP"]
USE_GRAD_SCALER = CONFIGS[CONFIG_ID]["USE_GRAD_SCALER"]
EARLY_STOP = CONFIGS[CONFIG_ID]["EARLY_STOP"]
PATIENCE = CONFIGS[CONFIG_ID]["PATIENCE"]
OPTIMIZER = CONFIGS[CONFIG_ID]["OPTIMIZER"]
OPTIMIZER_PARAMS = CONFIGS[CONFIG_ID]["OPTIMIZER_PARAMS"]
SCHEDULER = CONFIGS[CONFIG_ID]["SCHEDULER"]
SCHEDULER_PARAMS = CONFIGS[CONFIG_ID]["SCHEDULER_PARAMS"]
INPUT_SIZE = CONFIGS[CONFIG_ID]["INPUT_SIZE"]
NUM_WORKERS = CONFIGS[CONFIG_ID]["NUM_WORKERS"]
TRAIN_DIR = CONFIGS[CONFIG_ID]["TRAIN_DIR"] 
VAL_DIR = CONFIGS[CONFIG_ID]["VAL_DIR"]
TEST_DIR = CONFIGS[CONFIG_ID]["TEST_DIR"]
MODEL_FOLDER = Path(Path.cwd(), "models", str(CONFIGS[CONFIG_ID]["MODEL_NAME"]).split(".")[-1].split("'")[0])
ACCELERATOR = CONFIGS[CONFIG_ID]["ACCELERATOR"]
MATMUL_PRECISION = CONFIGS[CONFIG_ID]["MATMUL_PRECISION"]

ATF = {
    'aug': alb.Compose([
        alb.Resize(height=CONFIGS[CONFIG_ID]["INPUT_SIZE"][1], width=CONFIGS[CONFIG_ID]["INPUT_SIZE"][0]),
        alb.HorizontalFlip(),
        alb.VerticalFlip(),
    ]),
    'preproc': create_transform(**resolve_data_config({}, model=timm.create_model(CONFIGS[CONFIG_ID]["MODEL_NAME"], pretrained=CONFIGS[CONFIG_ID]["PRETRAINED"], num_classes=CONFIGS[CONFIG_ID]["NUM_CLASSES"]))),
    'resize_to_tensor': alb.Compose([
            alb.Resize(height=CONFIGS[CONFIG_ID]["INPUT_SIZE"][1], width=CONFIGS[CONFIG_ID]["INPUT_SIZE"][0]),
            ToTensorV2()
    ]),
}



