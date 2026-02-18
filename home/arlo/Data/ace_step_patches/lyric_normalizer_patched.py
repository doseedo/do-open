import re

# Fallback converters when OpenCC is not available
class FallbackConverter:
    """Simple fallback for OpenCC Traditional/Simplified conversion"""

    def __init__(self, conversion_type):
        self.conversion_type = conversion_type
        # Basic Traditional to Simplified mapping (common characters)
        self.t2s_map = {
            '繁': '繁', '體': '体', '學': '学', '語': '语', '時': '时',
            '個': '个', '進': '进', '來': '来', '開': '开', '關': '关',
            '與': '与', '過': '过', '這': '这', '說': '说', '會': '会',
            '現': '现', '國': '国', '見': '见', '經': '经', '還': '还',
            '給': '给', '聽': '听', '對': '对', '長': '长', '門': '门',
            '題': '题', '電': '电', '車': '车', '錢': '钱', '頭': '头'
        }
        # Reverse mapping for s2t
        self.s2t_map = {v: k for k, v in self.t2s_map.items()}

    def convert(self, text):
        if self.conversion_type == "t2s":
            for trad, simp in self.t2s_map.items():
                text = text.replace(trad, simp)
        elif self.conversion_type == "s2t":
            for simp, trad in self.s2t_map.items():
                text = text.replace(simp, trad)
        return text

# Try to import OpenCC, fallback to our implementation
try:
    from opencc import OpenCC
    t2s_converter = OpenCC("t2s")
    s2t_converter = OpenCC("s2t")
    OPENCC_AVAILABLE = True
except ImportError:
    print("⚠ OpenCC not available, using fallback Chinese conversion")
    t2s_converter = FallbackConverter("t2s")
    s2t_converter = FallbackConverter("s2t")
    OPENCC_AVAILABLE = False

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # Emoticons
    "]+",
    flags=re.UNICODE,
)

# 创建一个翻译表，用于替换和移除字符 (simplified from original)
TRANSLATION_TABLE = str.maketrans(
    {
        "-": " ",  # 将 '-' 替换为空格
        ",": None,
        ".": None,
        "，": None,
        "。": None,
        "!": None,
        "！": None,
        "?": None,
        "？": None,
        "…": None,
        ";": None,
        "；": None,
        ":": None,
        "：": None,
        "\u3000": " ",  # 将全角空格替换为空格
    }
)


def remove_emojis(text):
    return EMOJI_PATTERN.sub('', text)


def normalize_text(text, language="en", strip=True):
    """
    Normalize text for ACE-Step processing

    Args:
        text: Input text
        language: Language code (en, zh, etc.)
        strip: Whether to strip whitespace

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Remove emojis
    text = remove_emojis(text)

    # Language-specific processing
    if language in ["zh", "zh-cn", "zh-tw", "chinese"]:
        # Convert traditional to simplified Chinese
        text = t2s_converter.convert(text)
    elif language == "yue":  # Cantonese
        # Convert simplified to traditional for Cantonese
        text = s2t_converter.convert(text)

    # Apply character translation (remove punctuation)
    text = text.translate(TRANSLATION_TABLE)

    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    if strip:
        text = text.strip()

    return text


def preprocess_chinese_numbers(text):
    """Basic Chinese number preprocessing when full zh_num2words is not available"""
    # Simple replacements for common Chinese numbers
    chinese_nums = {
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
        '零': '0', '壹': '1', '贰': '2', '叁': '3', '肆': '4',
        '伍': '5', '陆': '6', '柒': '7', '捌': '8', '玖': '9',
        '拾': '10', '佰': '100', '仟': '1000', '万': '10000'
    }

    for chinese, arabic in chinese_nums.items():
        text = text.replace(chinese, arabic)

    return text