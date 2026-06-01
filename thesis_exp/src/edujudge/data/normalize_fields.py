"""Canonical EduBench mappings and conservative field normalization."""

from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Any

from thesis_exp.src.edujudge.data.reference_contract import EXPECTED_SUBJECTS
from thesis_exp.src.edujudge.utils.text_norm import detect_language_from_text, normalize_text, stringify


METRIC_SPECS: list[dict[str, str]] = [
    {
        "canonical_metric": "Instruction Following & Task Completion",
        "metric_abbr": "IFTC",
        "metric_group": "Scenario Adaptability",
        "zh": "指令遵循与任务完成",
    },
    {
        "canonical_metric": "Role & Tone Consistency",
        "metric_abbr": "RTC",
        "metric_group": "Scenario Adaptability",
        "zh": "角色与口吻一致性",
    },
    {
        "canonical_metric": "Content Relevance & Scope Control",
        "metric_abbr": "CRSC",
        "metric_group": "Scenario Adaptability",
        "zh": "内容相关性与范围控制",
    },
    {
        "canonical_metric": "Scenario Element Integration",
        "metric_abbr": "SEI",
        "metric_group": "Scenario Adaptability",
        "zh": "场景要素融合度",
    },
    {
        "canonical_metric": "Basic Factual Accuracy",
        "metric_abbr": "BFA",
        "metric_group": "Factual & Reasoning Accuracy",
        "zh": "基础事实准确性",
    },
    {
        "canonical_metric": "Domain Knowledge Accuracy",
        "metric_abbr": "DKA",
        "metric_group": "Factual & Reasoning Accuracy",
        "zh": "领域知识专业性",
    },
    {
        "canonical_metric": "Reasoning Process Rigor",
        "metric_abbr": "RPR",
        "metric_group": "Factual & Reasoning Accuracy",
        "zh": "推理过程严谨性",
    },
    {
        "canonical_metric": "Error Identification & Correction Precision",
        "metric_abbr": "EICP",
        "metric_group": "Factual & Reasoning Accuracy",
        "zh": "错误识别与纠正精确性",
    },
    {
        "canonical_metric": "Clarity, Simplicity & Inspiration",
        "metric_abbr": "CSI",
        "metric_group": "Pedagogical Application",
        "zh": "清晰易懂与表达启发",
    },
    {
        "canonical_metric": "Motivation, Guidance & Positive Feedback",
        "metric_abbr": "MGPF",
        "metric_group": "Pedagogical Application",
        "zh": "激励引导与积极反馈",
    },
    {
        "canonical_metric": "Personalization, Adaptation & Learning Support",
        "metric_abbr": "PALS",
        "metric_group": "Pedagogical Application",
        "zh": "个性化适应与学习支持",
    },
    {
        "canonical_metric": "Higher-Order Thinking & Skill Development",
        "metric_abbr": "HOTSD",
        "metric_group": "Pedagogical Application",
        "zh": "促进高阶思维与能力发展",
    },
]

METRIC_BY_CANONICAL = {m["canonical_metric"]: m for m in METRIC_SPECS}
METRIC_BY_ZH = {m["zh"]: m for m in METRIC_SPECS}
METRIC_BY_ABBR = {m["metric_abbr"].lower(): m for m in METRIC_SPECS}


SCENARIO_SPECS: list[dict[str, str]] = [
    {
        "canonical_scenario": "Question Answering",
        "scenario_abbr": "Q&A",
        "student_or_teacher_oriented": "Student-Oriented",
        "aliases": "question_answering|qa|q&a|problem_solving|答疑|回答问题",
    },
    {
        "canonical_scenario": "Error Correction",
        "scenario_abbr": "EC",
        "student_or_teacher_oriented": "Student-Oriented",
        "aliases": "error_correction|ec|纠错",
    },
    {
        "canonical_scenario": "Idea Provision",
        "scenario_abbr": "IP",
        "student_or_teacher_oriented": "Student-Oriented",
        "aliases": "idea_provision|ip|根据学生画像给出建议|建议生成|意见生成",
    },
    {
        "canonical_scenario": "Personalized Learning Support",
        "scenario_abbr": "PLS",
        "student_or_teacher_oriented": "Student-Oriented",
        "aliases": "personalized_learning_support|pls|根据学生画像设计学习路径|学习路径规划|个性化学习支持",
    },
    {
        "canonical_scenario": "Emotional Support",
        "scenario_abbr": "ES",
        "student_or_teacher_oriented": "Student-Oriented",
        "aliases": "psychological_support|emotional_support|es|学生心理健康判断与建议|心理健康|情感支持",
    },
    {
        "canonical_scenario": "Question Generation",
        "scenario_abbr": "QG",
        "student_or_teacher_oriented": "Teacher-Oriented",
        "aliases": "question_generation|qg|根据知识点生成问题|生成问题|出题",
    },
    {
        "canonical_scenario": "Automatic Grading",
        "scenario_abbr": "AG",
        "student_or_teacher_oriented": "Teacher-Oriented",
        "aliases": "automatic_grading|ag|判题|自动评分|批改",
    },
    {
        "canonical_scenario": "Teaching Material Generation",
        "scenario_abbr": "TMG",
        "student_or_teacher_oriented": "Teacher-Oriented",
        "aliases": "teaching_material_generation|tmg|教学素材生成|教学素材|教案",
    },
    {
        "canonical_scenario": "Personalized Content Creation",
        "scenario_abbr": "PCC",
        "student_or_teacher_oriented": "Teacher-Oriented",
        "aliases": "personalized_content_creation|pcc|个性化内容创建|个性化的学习内容或任务",
    },
]

SCENARIO_BY_CANONICAL = {s["canonical_scenario"]: s for s in SCENARIO_SPECS}


def _clean_label(value: object) -> str:
    text = stringify(value).strip()
    text = re.sub(r"^\s*\d+(?:\.\d+)?\s*[:：、.-]?\s*", "", text)
    return text.strip()


def _fold(value: object) -> str:
    return re.sub(r"[\s_/\-:：&，,()（）]+", "", _clean_label(value).lower())


def canonicalize_metric(raw_metric: object) -> dict[str, str]:
    raw = _clean_label(raw_metric)
    folded = _fold(raw)
    lang = detect_language_from_text(raw)
    if not raw:
        return {
            "canonical_metric": "unknown",
            "metric_abbr": "unknown",
            "metric_group": "unknown",
            "language": "unknown",
            "confidence": "none",
            "notes": "empty metric",
        }

    for spec in METRIC_SPECS:
        canonical = spec["canonical_metric"]
        zh = spec["zh"]
        if canonical.lower() in raw.lower() or zh in raw:
            return {**spec, "language": lang, "confidence": "high", "notes": "exact/contained official label"}
        if folded in {_fold(canonical), _fold(zh), _fold(spec["metric_abbr"])}:
            return {**spec, "language": lang, "confidence": "high", "notes": "normalized exact alias"}

    aliases = {
        "domainknowledgeprofessionalism": "Domain Knowledge Accuracy",
        "领域知识准确性": "Domain Knowledge Accuracy",
        "personalizedadaptationlearningsupport": "Personalization, Adaptation & Learning Support",
        "个性化适配与学习支持": "Personalization, Adaptation & Learning Support",
        "higherorderthinkingandskilldevelopment": "Higher-Order Thinking & Skill Development",
    }
    if folded in aliases:
        spec = METRIC_BY_CANONICAL[aliases[folded]]
        return {**spec, "language": lang, "confidence": "medium", "notes": "known wording variant"}

    return {
        "canonical_metric": "unknown",
        "metric_abbr": "unknown",
        "metric_group": "unknown",
        "language": lang,
        "confidence": "none",
        "notes": "no official metric alias matched",
    }


def canonicalize_scenario(raw_scenario: object) -> dict[str, str]:
    raw = _clean_label(raw_scenario)
    folded = _fold(raw)
    if not raw:
        return {
            "canonical_scenario": "unknown",
            "scenario_abbr": "unknown",
            "student_or_teacher_oriented": "unknown",
            "confidence": "none",
            "notes": "empty scenario",
        }
    for spec in SCENARIO_SPECS:
        aliases = [spec["canonical_scenario"], spec["scenario_abbr"], *spec["aliases"].split("|")]
        for alias in aliases:
            alias_folded = _fold(alias)
            if folded == alias_folded or (len(alias_folded) > 3 and alias in raw):
                notes = "matched official scenario alias"
                if folded in {"problemsolving", "problemsolving"} or normalize_text(raw) == "problem solving":
                    notes = "Problem Solving is treated as a PDF/local alias for official Question Answering"
                return {
                    "canonical_scenario": spec["canonical_scenario"],
                    "scenario_abbr": spec["scenario_abbr"],
                    "student_or_teacher_oriented": spec["student_or_teacher_oriented"],
                    "confidence": "high",
                    "notes": notes,
                }
            if len(alias_folded) > 3 and alias_folded in folded:
                return {
                    "canonical_scenario": spec["canonical_scenario"],
                    "scenario_abbr": spec["scenario_abbr"],
                    "student_or_teacher_oriented": spec["student_or_teacher_oriented"],
                    "confidence": "high",
                    "notes": "matched official scenario alias",
                }
    return {
        "canonical_scenario": "unknown",
        "scenario_abbr": "unknown",
        "student_or_teacher_oriented": "unknown",
        "confidence": "none",
        "notes": "no official scenario alias matched",
    }


def convert_score_to_five(raw_score: float) -> int | None:
    if raw_score is None:
        return None
    try:
        value = float(raw_score)
    except (TypeError, ValueError):
        return None
    if 1 <= value <= 2:
        return 1
    if 3 <= value <= 4:
        return 2
    if 5 <= value <= 6:
        return 3
    if 7 <= value <= 8:
        return 4
    if 9 <= value <= 10:
        return 5
    return None


def round_half_up(value: float) -> int:
    return int(float(value) + 0.5)


def detect_score_scale(scores: list[float]) -> str:
    valid = [float(s) for s in scores if isinstance(s, (int, float))]
    if not valid:
        return "unknown"
    if min(valid) >= 1 and max(valid) <= 5:
        return "1-5"
    if min(valid) >= 1 and max(valid) <= 10:
        return "1-10"
    return "unknown"


def extract_profile_dict(question: object) -> dict[str, Any]:
    text = stringify(question)
    candidates = []
    if "{" in text and "}" in text:
        start = text.find("{")
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
    for candidate in candidates:
        try:
            value = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return {}


SUBJECT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Applied Economics", ["applied economics", "policy evaluation", "应用经济", "政策评估"]),
    ("Aquaculture", ["aquaculture", "fish breeding", "marine aquaculture", "水产养殖", "鱼类繁殖"]),
    ("Automation", ["automation", "industrial automation", "自动化", "工业自动化"]),
    ("Basic Medicine", ["basic medicine", "molecular biology", "genetics", "基础医学", "解剖学", "病理学", "药理学"]),
    ("Biology", ["biology", "cellular", "ecology", "生物", "细胞", "生态"]),
    ("Business Administration", ["business administration", "business management", "工商管理", "商业管理", "战略管理", "市场营销"]),
    ("Chemistry", ["chemistry", "chemical", "organic chemistry", "化学", "有机化学"]),
    ("Chinese", ["chinese language", "chinese literature", "语文", "古诗词", "中文"]),
    ("Clinical Medicine", ["clinical medicine", "clinical diagnosis", "clinical skills", "临床医学", "临床诊断", "执业医师"]),
    ("Computer Science", ["computer science", "programming", "python", "algorithm", "coding", "计算机", "编程", "算法", "代码"]),
    ("Crop Science", ["crop science", "agronomy", "crop", "作物科学", "农学", "作物"]),
    ("English", ["english", "英语"]),
    ("General Pedagogy", ["general pedagogy", "pedagogy", "education", "educational equity", "教育学", "教育公平", "教学"]),
    ("Geography", ["geography", "map reading", "topographic", "地理", "地图"]),
    ("History", ["history", "historical", "历史", "世界史", "近代"]),
    ("Law", ["law", "legal", "criminal law", "法律", "法学", "刑法", "司法考试"]),
    ("Literature and Art", ["literature and art", "literature", "language arts", "art", "painting", "文学", "艺术", "写作", "阅读"]),
    ("Mathematics", ["math", "mathematics", "algebra", "geometry", "calculus", "statistics", "probability", "数学", "代数", "几何", "微积分", "统计", "概率"]),
    ("Military Science", ["military science", "military strategy", "defense", "军事", "军事战略", "战术", "无人机"]),
    ("Physical Education", ["physical education", "sports science", "biomechanics", "体育", "运动科学", "生物力学"]),
    ("Physics", ["physics", "mechanics", "electromagnetism", "物理", "力学", "电磁"]),
    ("Psychology", ["psychology", "mental health", "心理", "心理健康"]),
    ("Public Administration", ["public administration", "public policy", "policy analysis", "local government", "公共管理", "公共政策", "政策分析", "政府部门"]),
    ("Sociology", ["sociology", "social", "社会学", "社会"]),
    ("Theoretical Economics", ["theoretical economics", "economics", "economic", "econometric", "game theory", "microeconomics", "理论经济", "经济学", "博弈论", "微观经济"]),
]


SUBJECT_BY_FOLDED = {_fold(subject): subject for subject in EXPECTED_SUBJECTS}


def canonicalize_subject(raw_subject: object) -> dict[str, str]:
    raw = stringify(raw_subject).strip()
    folded = _fold(raw)
    if not raw:
        return {"canonical_subject": "unknown", "confidence": "none", "notes": "empty subject"}
    if folded in SUBJECT_BY_FOLDED:
        return {"canonical_subject": SUBJECT_BY_FOLDED[folded], "confidence": "high", "notes": "exact canonical subject"}
    for subject, keywords in SUBJECT_KEYWORDS:
        if subject not in EXPECTED_SUBJECTS:
            continue
        if folded == _fold(subject):
            return {"canonical_subject": subject, "confidence": "high", "notes": "normalized subject alias"}
        haystack = normalize_text(raw)
        for keyword in keywords:
            if normalize_text(keyword) in haystack:
                return {"canonical_subject": subject, "confidence": "medium", "notes": f"matched subject keyword: {keyword}"}
    return {"canonical_subject": "unknown", "confidence": "none", "notes": "no canonical subject alias matched"}


def extract_subject_from_question(question: object) -> tuple[str, str, str, str, str]:
    text = stringify(question)
    subject_match = re.search(r"(?im)^subject\s*:\s*([^\n\r]+)", text)
    if subject_match:
        raw = subject_match.group(1).strip()
        mapped = canonicalize_subject(raw)
        return raw, mapped["canonical_subject"], "question.Subject", "metadata_parse", mapped["confidence"]
    profile = extract_profile_dict(question)
    for key in ["subject", "Subject", "学科", "科目", "专业", "Major", "major"]:
        if key in profile:
            raw = stringify(profile.get(key))
            mapped = canonicalize_subject(raw)
            return raw, mapped["canonical_subject"], f"profile.{key}", "structured_field", mapped["confidence"]
    return "", "unknown", "", "", "none"


def infer_subject(question: object) -> tuple[str, str]:
    raw, canonical, _, _, _ = extract_subject_from_question(question)
    if canonical != "unknown":
        return raw, canonical
    text = stringify(question)
    profile = extract_profile_dict(question)
    values = " ".join(stringify(v) for v in profile.values())
    haystack = normalize_text(text + " " + values)
    hits: Counter[str] = Counter()
    raw_hit = ""
    for subject, keywords in SUBJECT_KEYWORDS:
        for keyword in keywords:
            if normalize_text(keyword) in haystack:
                hits[subject] += len(keyword)
                raw_hit = raw_hit or keyword
    if hits:
        subject = hits.most_common(1)[0][0]
        return raw_hit, subject
    return "unknown", "unknown"


def infer_education_level(question: object) -> tuple[str, str]:
    text = stringify(question)
    profile = extract_profile_dict(question)
    raw_candidates = []
    for key, value in profile.items():
        key_norm = normalize_text(key)
        if any(token in key_norm for token in ["grade", "education level", "年级", "学历", "学段"]):
            raw_candidates.append(stringify(value))
    raw = " | ".join(raw_candidates) if raw_candidates else ""
    search = normalize_text(text + " " + raw)

    grade_match = re.search(r"\bgrade['\" ]*[:=]?\s*(\d{1,2})\b", search)
    zh_grade_match = re.search(r"年级['\" ]*[:=：]?\s*(\d{1,2})", text)
    grade = None
    if grade_match:
        grade = int(grade_match.group(1))
    elif zh_grade_match:
        grade = int(zh_grade_match.group(1))

    if grade is not None:
        if grade <= 5:
            return raw or f"Grade {grade}", "elementary"
        if grade <= 8:
            return raw or f"Grade {grade}", "middle_school"
        return raw or f"Grade {grade}", "high_school"
    if any(k in search for k in ["elementary", "primary", "小学"]):
        return raw or "elementary", "elementary"
    if any(k in search for k in ["middle school", "junior high", "初中"]):
        return raw or "middle school", "middle_school"
    if any(k in search for k in ["high school", "高中"]):
        return raw or "high school", "high_school"
    if any(k in search for k in ["undergraduate", "college", "bachelor", "本科", "大学"]):
        return raw or "undergraduate", "undergraduate"
    if any(k in search for k in ["graduate", "master", "phd", "doctor", "硕士", "研究生", "博士"]):
        return raw or "graduate", "graduate"
    if any(k in search for k in ["adult", "professional", "teacher", "教师", "职场"]):
        return raw or "adult/professional", "adult_or_other"
    age_match = re.search(r"\bage['\" ]*[:=]?\s*(\d{1,2})\b", search)
    if age_match:
        age = int(age_match.group(1))
        if age <= 11:
            return f"Age {age}", "elementary"
        if age <= 14:
            return f"Age {age}", "middle_school"
        if age <= 18:
            return f"Age {age}", "high_school"
        if age <= 23:
            return f"Age {age}", "undergraduate"
        return f"Age {age}", "adult_or_other"
    return raw or "unknown", "unknown"


def infer_language(*values: object) -> str:
    text = "\n".join(stringify(v) for v in values)
    return detect_language_from_text(text)


def stable_language_label(language: str) -> str:
    """Use stable English labels for reports and plots."""
    if language == "zh":
        return "Chinese"
    if language == "en":
        return "English"
    return language or "unknown"


def stable_language_label(language: str) -> str:
    lang = normalize_text(language)
    if lang in {"en", "english"}:
        return "en"
    if lang in {"zh", "cn", "chinese", "中文", "汉语"}:
        return "zh"
    return language or "unknown"


def infer_field_candidates(keys: list[str], category: str) -> list[str]:
    patterns = {
        "score": ["score", "评分", "分数", "label", "human", "evaluate"],
        "question": ["question", "prompt", "instruction", "query", "题目", "问题"],
        "answer": ["answer", "response", "completion", "output", "回答", "答案"],
        "metric": ["metric", "principle", "criterion", "criteria", "维度", "指标"],
        "subject": ["subject", "discipline", "course", "knowledge", "学科", "知识点"],
        "scenario": ["scenario", "task", "scene", "场景", "任务"],
        "language": ["language", "lang", "语言"],
        "model": ["model", "generator", "answer_model", "模型"],
        "human_score": ["human", "annotator", "人工"],
        "judge_score": ["judge", "eval", "evaluator", "llm"],
    }
    wanted = patterns[category]
    out = []
    for key in keys:
        key_norm = normalize_text(key)
        if any(token in key_norm for token in wanted):
            out.append(key)
    return sorted(set(out))
