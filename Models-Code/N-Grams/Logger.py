import os
import logging
import sys

class Logger:

    logger_name = None
    log_file = None
    logger = None

    def __init__(self, log_file, logger_name, level):

        self.log_file = log_file
        self.level = level
        self.logger_name = logger_name


        try:
            self._setupLogger()
        except Exception:
            print('Error while setting up the logger!!')
            sys.exit(-1)


    def _setupLogger(self):

        self.logger = logging.getLogger(self.logger_name)
        formatter = logging.Formatter('%(asctime)s : %(message)s')
        fileHandler = logging.FileHandler(self.log_file, mode='a')
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler(sys.stdout)
        streamHandler.setLevel(100)
        streamHandler.setFormatter(formatter)

        self.logger.setLevel(self.level)
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(streamHandler)

    def getLogger(self):
        return self.logger

