# -*- coding: utf-8 -*-
"""
@Time: 2023/4/27 18:13 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import os
import logging


class MyLogger:
    def __init__(self, name, level=logging.INFO, log_file_path=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        self.formatter = logging.Formatter('%(message)s')

        if log_file_path is not None:
            log_dir = os.path.dirname(log_file_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)
