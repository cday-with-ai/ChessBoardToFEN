from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # API settings
    api_title: str = "Chess Position Recognition API"
    api_version: str = "1.0.0"
    api_description: str = "Converts chess board images to FEN notation"
    
    # Model settings
    model_path: Path = Path("models/piece_classifier.h5")
    confidence_threshold: float = 0.8
    
    # Image processing settings
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_types: set = {"image/jpeg", "image/png", "image/jpg"}
    
    # Board detection settings
    board_size: int = 8
    square_size: int = 64  # Size to resize each square for classification
    
    class Config:
        env_file = ".env"


settings = Settings()