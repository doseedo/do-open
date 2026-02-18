"""
Patched version of zh_num2words that works without OpenCC
"""

import re

class ZhNum2Words:
    """Fallback Chinese number to words converter"""

    def __init__(self, cc_mode="t2s"):
        self.cc_mode = cc_mode
        # Basic number mappings
        self.num_map = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
            '10': '十', '100': '百', '1000': '千', '10000': '万'
        }

    def convert(self, text):
        """Basic number to Chinese conversion"""
        # Simple digit replacement
        for digit, chinese in self.num_map.items():
            text = text.replace(digit, chinese)
        return text

def num_to_chinese(text):
    """Convert numbers to Chinese characters"""
    converter = ZhNum2Words()
    return converter.convert(text)

def chinese_to_num(text):
    """Convert Chinese numbers to digits"""
    chinese_nums = {
        '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
        '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        '十': '10', '百': '100', '千': '1000', '万': '10000'
    }

    for chinese, digit in chinese_nums.items():
        text = text.replace(chinese, digit)

    return text