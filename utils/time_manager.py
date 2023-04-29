# -*- coding: utf-8 -*-
"""
@Time: 2023/4/27 18:36 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import time


class MyTime:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        if self.end_time is None:
            raise ValueError("Timer has not been stopped.")
        elapsed_time_seconds = round(self.end_time - self.start_time, 2)

        elapsed_time_minutes = round(elapsed_time_seconds / 60.0, 2)
        return elapsed_time_seconds, elapsed_time_minutes


def get_format_time():
    """
    get the formatted current time

    :return:formatted_time
    """
    current_time = time.time()
    formatted_time = time.strftime("%m-%d %H%M%S", time.localtime(current_time))
    return formatted_time
