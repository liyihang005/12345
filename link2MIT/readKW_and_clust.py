# -*- coding:utf-8 -*-
# author: John Price
# datetime 2022 02 06
import pandas as pd
from worker import *

class Read(object):
    def __init__(self):
        pass

    @staticmethod
    def read_abs_kw_file(file_path, col_name):
        data = pd.read_excel(file_path)
        return data


