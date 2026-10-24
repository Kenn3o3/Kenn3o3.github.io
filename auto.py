import re

def convert_math(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 定义匹配模式，匹配不被双美元符号包围的单美元符号公式
    pattern = r'(?<!\$)\$(.+?)\$(?!\$)'

    # 使用替换函数，避免在替换字符串中处理转义字符
    def replacer(match):
        return '$$' + match.group(1) + '$$'

    # 执行替换
    new_content = re.sub(pattern, replacer, content)

    # 将结果写回文件
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(new_content)

# 示例用法：
convert_math(r'C:\Users\user\Desktop\Kenn3o3.github.io\_posts\2024-02-17-3D-Reconstruction.md')
