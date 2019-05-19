''' Parameters for Training '''
IS_TRAIN=True

NUM_EPOCHS=200
INITIAL_LEARNING_RATE=3e-4  # 0.0003
BATCH_SIZE=2
NUM_FRAMES_PER_CLIP=8

LATENT_DIM=128

IMAGE_CROP_SIZE=224

MODEL_SAVE_INTERVAL_EPOCH=1

MODEL_DIR="./model"
RESULT_DIR="./result"
LOG_DIR="./log"

''' Datasets Path '''
BASE_DATASET_PATH="/Users/kimsu/Desktop/kimsu/01_study/05_src/anomaly_detection/dataset"
UCSD_DATASET_PATH="/Users/kimsu/Desktop/kimsu/01_study/05_src/anomaly_detection/dataset/UCSD_Anomaly_Dataset.v1p2"
AVENUE_DATASET_PATH="/Users/kimsu/Desktop/kimsu/01_study/05_src/anomaly_detection/dataset/Avenue_Dataset"
AMITSUBWAY_DATASET_PATH="/Users/kimsu/Desktop/kimsu/01_study/05_src/anomaly_detection/dataset/Amit_Subway"