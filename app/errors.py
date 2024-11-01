from enum import Enum,  auto
class ErrorType(Enum):
    API_KEY_ERROR = auto()
    OTHER_ERRORS = auto()


ERROR_LIST = [ErrorType.API_KEY_ERROR, 
              ErrorType.OTHER_ERRORS]