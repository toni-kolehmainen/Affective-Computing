from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MELD_DATASET_PATH: str = (
        "PATH-TO-FOLDER-WITH-MELD"
    )
    EMOTIC_DATASET_PATH: str = (
        "PATH-TO-FOLDER-WITH-EMOTIC"
    )
    EMOTIONCLIP_MODEL_PATH: str = "PATH-TO-EMOTIONCLIP-MODEL"

    BACKBONE_BASE_PATH: str = "./src/models/model_configs/ViT-B-32.json"


settings = Settings()
