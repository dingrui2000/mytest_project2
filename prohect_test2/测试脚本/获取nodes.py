import csv

input_file = r'F:\learningfiles\myproject\congrat-master\prohect_test2\data\大学课程数据集（概念+描述+嵌入）.csv'
output_file = 'node.csv'

with open(input_file, 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)

        for row in reader:
            # 保留第一列，删除第二列，并将第三列移动到第二列的位置
            new_row = [row[0], row[2]]
            writer.writerow(new_row)

print(f"{output_file} has been generated!")
