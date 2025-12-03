"""
替换判题任务数据:
- processed_excel_data_2_zh.jsonl 替换 5_merge_human_metric_zh.jsonl 中的判题任务
- processed_excel_data_2_en.jsonl 替换 5_merge_human_metric_en.jsonl 中的判题任务

"判题"任务的特征：question 字段包含 "请根据问题和学生答案给出"
"""

import json
import os


def load_jsonl(filepath):
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, filepath):
    """保存 JSONL 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def is_grading_task_zh(entry):
    """判断是否为中文"判题"任务"""
    question = entry.get('question', '')
    return '请根据问题和学生答案给出' in question


def is_grading_task_en(entry):
    """判断是否为英文"判题"任务"""
    question = entry.get('question', '')
    # 英文判题任务的特征
    return 'Return the result in JSON format' in question and 'Score' in question and 'Personalized Feedback' in question


def replace_grading_data(original_file, new_file, output_file, is_grading_func, lang):
    """替换判题数据"""
    print(f"\n{'='*50}")
    print(f"处理 {lang} 数据...")
    print(f"{'='*50}")
    
    # 加载数据
    print(f"加载原始数据: {original_file}")
    original_data = load_jsonl(original_file)
    print(f"原始数据条数: {len(original_data)}")
    
    print(f"加载新标注数据: {new_file}")
    new_data = load_jsonl(new_file)
    print(f"新标注数据条数: {len(new_data)}")
    
    # 统计原始数据中的判题任务数量
    grading_count = sum(1 for entry in original_data if is_grading_func(entry))
    non_grading_count = len(original_data) - grading_count
    print(f"\n原始数据中判题任务数量: {grading_count}")
    print(f"原始数据中非判题任务数量: {non_grading_count}")
    
    # 分离非判题任务
    non_grading_data = [entry for entry in original_data if not is_grading_func(entry)]
    
    # 构建替换后的数据：非判题任务 + 新标注的判题数据
    replaced_data = non_grading_data + new_data
    
    print(f"\n替换后数据总数: {len(replaced_data)}")
    print(f"  - 保留的非判题任务: {len(non_grading_data)}")
    print(f"  - 新增的判题任务: {len(new_data)}")
    
    # 保存替换后的数据
    save_jsonl(replaced_data, output_file)
    print(f"已保存到: {output_file}")
    
    return replaced_data


def print_statistics(data, title):
    """打印统计信息"""
    print(f"\n=== {title} ===")
    
    # 按 principle 统计
    principle_counts = {}
    for entry in data:
        principle = entry.get('principle', 'unknown')
        principle_counts[principle] = principle_counts.get(principle, 0) + 1
    
    print("\n按评估指标统计:")
    for principle, count in sorted(principle_counts.items(), key=lambda x: -x[1]):
        print(f"  {principle}: {count}")
    
    # 按 model 统计
    model_counts = {}
    for entry in data:
        model = entry.get('model', 'unknown')
        model_counts[model] = model_counts.get(model, 0) + 1
    
    print("\n按模型统计:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"  {model}: {count}")
    
    # 按分数统计
    score_counts = {}
    for entry in data:
        score = entry.get('score', 'unknown')
        score_counts[score] = score_counts.get(score, 0) + 1
    
    print("\n按分数统计:")
    for score, count in sorted(score_counts.items(), key=lambda x: str(x[0])):
        print(f"  {score}分: {count}")


def main():
    # 处理中文数据
    zh_replaced = replace_grading_data(
        original_file='5-grades/5_merge_human_metric_zh.jsonl',
        new_file='deepseek_output/processed_excel_data_2_zh.jsonl',
        output_file='5-grades/5_merge_human_metric_zh_replaced.jsonl',
        is_grading_func=is_grading_task_zh,
        lang='中文'
    )
    
    # 处理英文数据
    en_replaced = replace_grading_data(
        original_file='5-grades/5_merge_human_metric_en.jsonl',
        new_file='deepseek_output/processed_excel_data_2_en.jsonl',
        output_file='5-grades/5_merge_human_metric_en_replaced.jsonl',
        is_grading_func=is_grading_task_en,
        lang='英文'
    )
    
    # 打印统计
    print_statistics(zh_replaced, "中文数据统计")
    print_statistics(en_replaced, "英文数据统计")


if __name__ == "__main__":
    main()
