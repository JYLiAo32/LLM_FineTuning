import colorama
from colorama import Fore, Style, Back

red_prefix = Fore.RED
gre_prefix = Fore.GREEN
light_prefix = Fore.LIGHTWHITE_EX
default_color = '\033[0m'


def color_text(color, text):
    color_code = COLOR_MAP.get(color.lower(), "")
    return f"{color_code}{text}{Style.RESET_ALL}"


colorama.init(autoreset=True)  # autoreset=True 确保每次打印后自动重置颜色

# 颜色映射表：键为用户友好的颜色名，值为 colorama 的颜色常量
COLOR_MAP = {
    "black": Fore.BLACK,
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
    "white": Fore.WHITE,
    "reset": Fore.RESET,
    ###########
    "warning": Fore.YELLOW,
    "note": Fore.BLUE,
    "error": Fore.RED,
}


def colored_print(text: str, color: str = None, bold: bool = False) -> None:
    color_code = COLOR_MAP.get(color.lower(), "") if color else ""
    style_code = Style.BRIGHT if bold else ""
    formatted_text = f"{style_code}{color_code}{text}{Style.RESET_ALL}"
    print(formatted_text)