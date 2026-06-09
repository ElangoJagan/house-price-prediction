import sys
from src.logger import logger


class CustomException(Exception):
    """
    Custom exception class that captures file name,
    line number and error message for better debugging.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self._get_detailed_error(error_message, error_detail)
        logger.error(self.error_message)

    @staticmethod
    def _get_detailed_error(error_message, error_detail: sys) -> str:
        """Extracts file name and line number from traceback."""
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return (
            f"Error occurred in file [ {file_name} ] "
            f"line number [ {line_number} ] "
            f"error message [ {str(error_message)} ]"
        )

    def __str__(self):
        return self.error_message