import csv

# 读取原始数据集
with open(r'F:\learningfiles\myproject\congrat-master\prohect_test2\data\概念依赖关系（大学课程数据集En）.csv', 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    next(reader)  # 跳过标题行

    # 创建新的edge.csv数据集
    with open('edge.csv', 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Source', 'Target'])  # 写入标题行

        for row in reader:
            # 只取前三个值
            concept_a, concept_b, dependency = row[:3]
            if int(dependency) == 1:
                print(f"Writing edge: {concept_a} -> {concept_b}")  # 打印正在写入的边
                writer.writerow([concept_a, concept_b])

print("edge.csv has been generated!")