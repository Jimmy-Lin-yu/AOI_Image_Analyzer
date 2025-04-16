import re

class FilenameTranslator:
    def __init__(self):
        self.translation_map = {
            "全色域环形100": "full_color_ring_100",
            "全色域条形30100": "full_color_bar_30100",
            "全色域同轴60": "full_color_coaxial_60",
            "4层4角度全圆88":"4_layer_4_angle_full_circle_88",
            "全色域圆顶":"full_color_dome",
            "亮度": "brightness",
            "白": "W",
            "红": "R",
            "绿": "G",
            "蓝": "B",
            "高": "height",
            "曝光": "exposure"
        }

    def translate(self, filename: str) -> str:
        # Replace each Chinese term with its English equivalent
        translated = filename
        for zh, en in self.translation_map.items():
            translated = translated.replace(zh, en)
        return translated

# Example usage
if __name__ == "__main__":
    filename = "112605_全色均環形100_亮度D1024白W0紅R0綠G1000藍B0_高H0曝光E1000S18.bmp"
    translator = FilenameTranslator()
    result = translator.translate(filename)
    print("Original: ", filename)
    print("Translated: ", result)
