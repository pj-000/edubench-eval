"""
替换 human_sampled_eval_sft.json 中的"判题"任务数据
用新标注的数据 (processed_excel_data_2_zh.jsonl) 进行替换

要求：
1. 删除原有包含"请根据问题和学生答案给出"评分":"评分细节":"个性化反馈""的判题任务
2. 用新标注数据替换，使用 deepseek_judge_only.py 的 prompt_template_zh 格式作为对话内容
3. 保留"我将向你提供一段教育领域下特定场景的对话"的框架
4. output 字段模仿原有的多指标评分格式
"""

import json
import re


# deepseek_judge_only.py 中的 prompt 模板（中文版）
PROMPT_TEMPLATE_ZH = """你需要实现：
1. 针对题目的评分细则和学生回答，生成评分和评分细节。
2. 针对学生答题情况生成具体、有建设性的反馈意见，例如可能涉及的知识盲区，学习建议等，语言积极、富有建设性。

学科：{subject}
教育阶段：{level}
题目类型：{question_type}
问题：{question}
标准答案：{standard_answer}
评分细则：{grading_criteria}
学生的答案：{student_answer}

以json格式返回
"评分":""
"评分细节":""
"个性化反馈":""
"""


def load_sft_data(filepath):
    """加载 SFT 训练数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_new_annotations(filepath):
    """加载新标注的数据"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def is_grading_task(entry):
    """判断是否为"判题"任务（需要删除的）"""
    instruction = entry.get('instruction', '')
    # 检测对话中是否包含判题任务的特征模式
    # 这些是原有的判题任务，user 要求模型给出"评分"/"评分细节"/"个性化反馈"
    return '请根据问题和学生答案给出' in instruction and '评分细节' in instruction and '个性化反馈' in instruction


def parse_question_fields(question_text):
    """
    从 question 文本中解析出各个字段
    question 格式示例：
    你需要完成以下任务：...
    Subject: 心理学
    Level: 博士
    QuestionType: 选择题
    Question: xxx
    StandardAnswer: A
    GradingCriteria: {...}
    StudentAnswer: C
    请以JSON格式返回结果：...
    """
    fields = {}
    
    # 使用正则表达式提取各字段
    patterns = {
        'subject': r'Subject:\s*(.+?)(?:\n|$)',
        'level': r'Level:\s*(.+?)(?:\n|$)',
        'question_type': r'QuestionType:\s*(.+?)(?:\n|$)',
        'question': r'Question:\s*(.+?)(?=\nStandardAnswer:)',
        'standard_answer': r'StandardAnswer:\s*(.+?)(?=\nGradingCriteria:)',
        'grading_criteria': r'GradingCriteria:\s*(.+?)(?=\nStudentAnswer:)',
        'student_answer': r'StudentAnswer:\s*(.+?)(?=\n\n请以JSON格式|\n请以JSON格式|$)',
    }
    
    for field, pattern in patterns.items():
        match = re.search(pattern, question_text, re.DOTALL)
        if match:
            fields[field] = match.group(1).strip()
        else:
            fields[field] = ''
    
    return fields


def escape_for_dialogue(s):
    """转义字符串，用于嵌入到对话表示中"""
    s = s.replace('\\', '\\\\')
    s = s.replace('\n', '\\n')
    s = s.replace('\r', '\\r')
    s = s.replace('\t', '\\t')
    s = s.replace("'", "\\'")
    s = s.replace('"', '\\"')
    return s


def convert_new_data_to_sft(new_entry):
    """
    将新标注数据转换为 SFT 格式
    
    新数据格式 (processed_excel_data_2_zh.jsonl):
    - question: 包含英文字段名的 prompt（需要解析出字段后用中文模板重新拼接）
    - response: 模型对学生作答的评分回复 (Score/ScoringDetails/PersonalizedFeedback)
    - principle/score/reason: 人工标注的评估结果（单个指标）
    
    目标格式:
    - instruction: "我将向你提供一段教育领域下特定场景的对话..." + 对话内容 + 评估指标
    - input: ""
    - output: 评估结果 JSON (单个指标)
    """
    question_text = new_entry.get('question', '')
    response = new_entry.get('response', '')
    principle = new_entry.get('principle', '')
    score = new_entry.get('score', '')
    reason = new_entry.get('reason', '')
    
    # 从 question 中解析出各个字段
    fields = parse_question_fields(question_text)
    
    # 使用 deepseek_judge_only.py 的中文 prompt 模板重新拼接
    user_content = PROMPT_TEMPLATE_ZH.format(
        subject=fields.get('subject', ''),
        level=fields.get('level', ''),
        question_type=fields.get('question_type', ''),
        question=fields.get('question', ''),
        standard_answer=fields.get('standard_answer', ''),
        grading_criteria=fields.get('grading_criteria', ''),
        student_answer=fields.get('student_answer', '')
    )
    
    # 转义字符串用于对话
    user_content_escaped = escape_for_dialogue(user_content)
    response_escaped = escape_for_dialogue(response)
    
    # 构建 instruction：保留开头框架 + 对话（使用中文 prompt 模板）+ 评估指标
    instruction = f"""我将向你提供一段教育领域下特定场景的对话，请根据所给定的所有评估指标及其评分细则对所给的回答进行评分并给出原因。
以JSON的格式返回，例如：
```json[{{"criterion": "<评估指标1名称>", "score": <得分>, "reason": "<原因>"}}, {{"criterion": "<评估指标2名称>", "score": <得分>, "reason": "<原因>"}}, ...]```

对话：
[{{'role': 'user', 'content': '{user_content_escaped}'}}, {{'role': 'assistant', 'content': '{response_escaped}'}}]
评估指标: 
[{{'metric': '{principle}', 'description': '根据评分细则评估学生作答情况', 'levels': ['5分：完全符合要求', '4分：基本符合要求', '3分：部分符合要求', '2分：不太符合要求', '1分：完全不符合要求']}}]"""

    # 构建 output - 模仿原有格式（单个指标的评分）
    reason_escaped = reason.replace('"', '\\"')
    output = f'```json[{{"criterion": "{principle}", "score": {score}, "reason": "{reason_escaped}"}}]```'
    
    return {
        "instruction": instruction,
        "input": "",
        "output": output
    }


def main():
    # 文件路径
    sft_filepath = r"e:\Daily Life\edubench-eval\train\human_sampled_eval_sft.json"
    new_data_filepath = r"e:\Daily Life\edubench-eval\deepseek_output\processed_excel_data_2_zh.jsonl"
    output_filepath = r"e:\Daily Life\edubench-eval\train\human_sampled_eval_sft_replaced.json"
    
    # 加载数据
    print("加载 SFT 数据...")
    sft_data = load_sft_data(sft_filepath)
    print(f"原 SFT 数据条目数: {len(sft_data)}")
    
    print("加载新标注数据...")
    new_annotations = load_new_annotations(new_data_filepath)
    print(f"新标注数据条目数: {len(new_annotations)}")
    
    # 统计并移除"判题"任务
    grading_count = 0
    non_grading_entries = []
    
    for entry in sft_data:
        if is_grading_task(entry):
            grading_count += 1
        else:
            non_grading_entries.append(entry)
    
    print(f"识别到的'判题'任务数量: {grading_count}")
    print(f"保留的非'判题'任务数量: {len(non_grading_entries)}")
    
    # 转换新标注数据为 SFT 格式
    print("转换新标注数据为 SFT 格式...")
    new_sft_entries = []
    for entry in new_annotations:
        new_sft_entry = convert_new_data_to_sft(entry)
        new_sft_entries.append(new_sft_entry)
    
    print(f"转换后的新 SFT 条目数: {len(new_sft_entries)}")
    
    # 合并数据：保留的非判题任务 + 新的判题任务
    final_data = non_grading_entries + new_sft_entries
    print(f"最终 SFT 数据条目数: {len(final_data)}")
    
    # 保存结果
    print(f"保存到 {output_filepath}...")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    
    print("完成！")
    
    # 显示一个转换后的示例
    if new_sft_entries:
        print("\n=== 转换后的示例 ===")
        sample = new_sft_entries[0]
        print(f"instruction 长度: {len(sample['instruction'])}")
        print(f"output: {sample['output'][:200]}...")


if __name__ == "__main__":
    main()
