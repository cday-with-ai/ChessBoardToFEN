class ChessRecognitionException(Exception):
    """Base exception for chess recognition errors"""
    pass


class BoardDetectionError(ChessRecognitionException):
    """Raised when board cannot be detected in the image"""
    pass


class InvalidImageError(ChessRecognitionException):
    """Raised when image is invalid or corrupt"""
    pass


class ModelNotFoundError(ChessRecognitionException):
    """Raised when the ML model file is not found"""
    pass