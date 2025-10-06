from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MELD_DATASET_PATH: str = (
        "/Data/Documents/masters/uni_oulu/semester1/affective_computing/project/MELD.Raw"
    )
    EMOTIC_DATASET_PATH: str = (
        "/Data/Documents/masters/uni_oulu/semester1/affective_computing/project/emotic"
    )
    
    EMOTIONCLIP_MODEL_PATH: str = './emotionclip_latest.pt'

    BACKBONE_BASE_PATH: str = "./src/models/model_configs/ViT-B-32.json"


settings = Settings()