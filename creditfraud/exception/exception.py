from creditfraud.logging.logger import logging
import sys

class CreditFraudException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is not None:
            self.lineno = exc_tb.tb_lineno
            self.filename = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = "N/A"
            self.filename = "N/A"

        self.error_message = error_message

    def __str__(self):
        return (
            f"Error occurred in python script [{self.filename}] "
            f"at line number [{self.lineno}] error message [{self.error_message}]"
        )

    
# just testing the exception class
if __name__ == "__main__":
    try:
        logging.info(" This is an info message entering try block")
        a = 1/0
        print("testing exception")
    except Exception as e:
        raise CreditFraudException(e, sys)