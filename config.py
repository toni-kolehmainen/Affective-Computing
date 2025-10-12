from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    CUDA_DEVICE: str = "cuda:0"

    EMOTION_LIST: list[str] = ["neutral", "joy", "sadness", "anger", "fear", "surprise"]

    MELD_DATASET_PATH: str = "./project/MELD.Raw"
    EMOTIC_DATASET_PATH: str = "./project/emotic"
    EMOTIONCLIP_MODEL_PATH: str = "./data/emotionclip_latest.pt"

    FRAMES_ROOT: str = "./project/MELD.Raw/test/frames/"
    BBOX_JSON: str = "./project/MELD.Raw/test/human_boxes.json"
    TEST_CSV: str = "./project/MELD.Raw/test/test_sent_emo.csv"

    BACKBONE_BASE_PATH: str = "./src/models/model_configs/ViT-B-32.json"


settings = Settings()
