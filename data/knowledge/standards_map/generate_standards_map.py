import pandas as pd
import json

# 读取 Excel 文件
file_path = r'C:\Users\maoqr\Desktop\embedding\data-prot\src\trandatacls\zq\data\关基-数据分类分级目录.xlsx'  # 替换为你的 Excel 文件路径
df = pd.read_excel(file_path, header=1)  # 忽略第一行，从第二行开始读取

# 初始化一个空字典，用于存储处理后的数据
result_data = {}

# 遍历每一行数据
for index, row in df.iterrows():
    # 获取当前行的“定义说明4”和“四级分类”
    definition_4 = row['定义说明4']
    fourth_category = row['四级分类']

    # 如果“定义说明4”为"——"，则使用“定义说明3”
    if definition_4 == '——':
        definition_4 = row['定义说明3']

    # 如果“四级分类”为"——"，则使用“三级分类”
    if fourth_category == '——':
        fourth_category = row['三级分类']

    # 删除 category 中的换行符
    fourth_category = str(fourth_category).replace('\n', '')

    # 将处理后的数据添加到结果字典中
    result_data[definition_4] = {
        "category": fourth_category
    }

# 将结果数据转换为 JSON 格式
result_json = json.dumps(result_data, ensure_ascii=False, indent=4)

# 打印 JSON 数据
print(result_json)

# 将结果保存为 JSON 文件
output_file = 'guanji_dict.json'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(result_json)

print(f"数据已保存到 {output_file}")