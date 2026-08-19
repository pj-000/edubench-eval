#!/usr/bin/env python3
"""Generate reader-facing LaTeX tables from frozen HMSA experiment data."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "tables"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write(stem: str, text: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{stem}.tex").write_text(text.strip() + "\n", encoding="utf-8")


def pm(value: float, deviation: float, digits: int = 3) -> str:
    return f"{value:.{digits}f} $\\pm$ {deviation:.{digits}f}"


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def methods_table() -> None:
    text = r"""
\begin{table}[H]
\centering
\caption{六种监督配置。Direct-soft 与 Single-head mix 为随机种子 42 的诊断配置；Shuffled-soft 与 MeanAux 为事后开发集机制对照。}
\label{tab:method_matrix}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\linewidth}{@{}lYYcYcY@{}}
\toprule
配置 & 主头目标 & 辅助头目标 & 头数 & 推理 & 种子 & 证据角色 \\
\midrule
Hard-only & 聚合硬标签 & -- & 1 & 硬头 & 42/43/44 & 匹配基线 \\
Direct-soft & 三人经验分布 & -- & 1 & 同一头 & 42 & 诊断实验 \\
Single-head mix & 硬/软目标各 0.5 & -- & 1 & 同一头 & 42 & 诊断实验 \\
Shuffled-soft & 聚合硬标签 & 标签内置换 & 2 & 仅硬头 & 42/43/44 & 事后机制对照 \\
MeanAux & 聚合硬标签 & 连续人类均分 & 2 & 仅硬头 & 42/43/44 & 事后目标对照 \\
HMSA & 聚合硬标签 & 三人经验分布 & 2 & 仅硬头 & 42/43/44 & 三种子评估 \\
\bottomrule
\end{tabularx}
\end{table}
"""
    write("tab_method_matrix", text)


def positioning_table() -> None:
    text = r"""
\begin{table}[H]
\centering
\caption{本文与 Fornaciari 等（2021）soft-label multi-task 范式的定位比较。}
\label{tab:positioning}
\small
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.16\linewidth}YY@{}}
\toprule
维度 & Fornaciari 等（2021） & 本文 \\
\midrule
共同结构 & 硬标签主任务与分布辅助任务共享编码器 & 沿用硬主头--软辅助头共享主干 \\
任务设置 & 词性标注、词干分类 & 教育回答 1--5 级序数评分 \\
核心问题 & 利用标签分歧改进预测 & 比较三种软监督放置方式 \\
损失设计 & 主任务与辅助分布损失等权联合 & 硬标签 CE 与分布 CE 等权联合 \\
评分接口 & 主任务头输出标签 & 仅由硬头输出整数分数 \\
证据重点 & 2 项 NLP 任务、3 个数据集、3 种辅助损失 & 序数指标、风险分层、置换与均分对照 \\
\bottomrule
\end{tabularx}
\end{table}
"""
    write("tab_positioning", text)


def protocol_table() -> None:
    dataset = {row["split"]: row for row in read_csv(DATA / "dataset_summary.csv")}
    text = rf"""
\begin{{table}}[H]
\centering
\caption{{固定数据划分及硬标签分布。}}
\label{{tab:data_distribution}}
\small
\begin{{tabular}}{{lrrrrrr}}
\toprule
划分 & $n$ & 1 分 & 2 分 & 3 分 & 4 分 & 5 分 \\
\midrule
训练集 & {int(dataset['train']['n']):,} & {int(dataset['train']['label_1']):,} & {int(dataset['train']['label_2']):,} & {int(dataset['train']['label_3']):,} & {int(dataset['train']['label_4']):,} & {int(dataset['train']['label_5']):,} \\
开发集 & {int(dataset['dev']['n']):,} & {int(dataset['dev']['label_1']):,} & {int(dataset['dev']['label_2']):,} & {int(dataset['dev']['label_3']):,} & {int(dataset['dev']['label_4']):,} & {int(dataset['dev']['label_5']):,} \\
测试集 & {int(dataset['test']['n']):,} & {int(dataset['test']['label_1']):,} & {int(dataset['test']['label_2']):,} & {int(dataset['test']['label_3']):,} & {int(dataset['test']['label_4']):,} & {int(dataset['test']['label_5']):,} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write("tab_data_protocol", text)


def main_results_table() -> None:
    test_rows = {row["metric"]: row for row in read_csv(DATA / "test_main_results.csv")}
    dev_rows = {row["group"]: row for row in read_csv(DATA / "dev_disagreement_summary.csv")}
    dev = dev_rows["all"]
    formal = read_json(DATA / "formal_decision.json")

    dev_values = {
        "MAE": (float(dev["b0_MAE_human_mean_mean"]), float(dev["b0_MAE_human_mean_sd"]), float(dev["exp51_MAE_human_mean_mean"]), float(dev["exp51_MAE_human_mean_sd"]), float(dev["delta_MAE_human_mean_mean"])),
        "Exact": (float(dev["b0_Exact_rounded_mean"]), float(dev["b0_Exact_rounded_sd"]), float(dev["exp51_Exact_rounded_mean"]), float(dev["exp51_Exact_rounded_sd"]), float(dev["delta_Exact_rounded_mean"])),
        "Kendall": (formal["mean"]["b0_kendall"], formal["standard_deviation"]["b0_kendall"], formal["mean"]["exp51_kendall"], formal["standard_deviation"]["exp51_kendall"], formal["mean"]["delta_kendall"]),
        "Bias": (float(dev["b0_Bias_human_mean_mean"]), float(dev["b0_Bias_human_mean_sd"]), float(dev["exp51_Bias_human_mean_mean"]), float(dev["exp51_Bias_human_mean_sd"]), float(dev["delta_Bias_human_mean_mean"])),
        "L2H": (formal["mean"]["b0_l2h_count"], formal["standard_deviation"]["b0_l2h_count"], formal["mean"]["exp51_l2h_count"], formal["standard_deviation"]["exp51_l2h_count"], formal["mean"]["delta_l2h_count"]),
        "QWK": (float(dev["b0_QWK_rounded_mean"]), float(dev["b0_QWK_rounded_sd"]), float(dev["exp51_QWK_rounded_mean"]), float(dev["exp51_QWK_rounded_sd"]), float(dev["delta_QWK_rounded_mean"])),
    }
    mapping = {
        "MAE": "MAE_human_mean",
        "Exact": "Exact_rounded",
        "Kendall": "Kendall_human_mean",
        "Bias": "Bias_human_mean",
        "L2H": "L2H_count",
        "QWK": "QWK_rounded",
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Hard-only 与 HMSA 的三种子结果（均值 $\pm$ 样本标准差）。所有差值统一为 HMSA$-$Hard-only。}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"阶段 & 指标 & Hard-only & HMSA & $\Delta$ \\",
        r"\midrule",
    ]
    for index, metric in enumerate(("MAE", "Exact", "Kendall", "Bias", "L2H", "QWK")):
        b0, b0_sd, exp, exp_sd, delta = dev_values[metric]
        stage = "Formal dev" if index == 0 else ""
        lines.append(f"{stage} & {metric} & {pm(b0, b0_sd)} & \\textbf{{{pm(exp, exp_sd)}}} & {signed(delta)} \\\\")
    lines.append(r"\midrule")
    for index, metric in enumerate(("MAE", "Exact", "Kendall", "Bias", "L2H", "QWK")):
        row = test_rows[mapping[metric]]
        stage = "Frozen test" if index == 0 else ""
        lines.append(
            f"{stage} & {metric} & {pm(float(row['b0_mean']), float(row['b0_sd']))} & "
            f"\\textbf{{{pm(float(row['exp51_mean']), float(row['exp51_sd']))}}} & {signed(float(row['delta_mean']))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    write("tab_main_results", "\n".join(lines))


def main_results_table_v2() -> None:
    """Write frozen-test results for the main text and formal-dev results for the appendix."""
    test_rows = {row["metric"]: row for row in read_csv(DATA / "test_main_results.csv")}
    dev_rows = {row["group"]: row for row in read_csv(DATA / "dev_disagreement_summary.csv")}
    dev = dev_rows["all"]
    formal = read_json(DATA / "formal_decision.json")
    mapping = {
        "MAE": "MAE_human_mean",
        "Exact": "Exact_rounded",
        "Kendall": "Kendall_human_mean",
        "Bias": "Bias_human_mean",
        "L2H": "L2H_count",
        "QWK": "QWK_rounded",
    }
    display_names = {
        "MAE": r"MAE $\downarrow$",
        "Exact": r"精确匹配率 $\uparrow$",
        "Kendall": r"Kendall $\tau_b$ $\uparrow$",
        "Bias": r"有符号偏差（0 最佳）",
        "L2H": r"L2H 数量 $\downarrow$",
        "QWK": r"QWK $\uparrow$",
    }
    dev_values = {
        "MAE": (float(dev["b0_MAE_human_mean_mean"]), float(dev["b0_MAE_human_mean_sd"]), float(dev["exp51_MAE_human_mean_mean"]), float(dev["exp51_MAE_human_mean_sd"]), float(dev["delta_MAE_human_mean_mean"])),
        "Exact": (float(dev["b0_Exact_rounded_mean"]), float(dev["b0_Exact_rounded_sd"]), float(dev["exp51_Exact_rounded_mean"]), float(dev["exp51_Exact_rounded_sd"]), float(dev["delta_Exact_rounded_mean"])),
        "Kendall": (formal["mean"]["b0_kendall"], formal["standard_deviation"]["b0_kendall"], formal["mean"]["exp51_kendall"], formal["standard_deviation"]["exp51_kendall"], formal["mean"]["delta_kendall"]),
        "Bias": (float(dev["b0_Bias_human_mean_mean"]), float(dev["b0_Bias_human_mean_sd"]), float(dev["exp51_Bias_human_mean_mean"]), float(dev["exp51_Bias_human_mean_sd"]), float(dev["delta_Bias_human_mean_mean"])),
        "L2H": (formal["mean"]["b0_l2h_count"], formal["standard_deviation"]["b0_l2h_count"], formal["mean"]["exp51_l2h_count"], formal["standard_deviation"]["exp51_l2h_count"], formal["mean"]["delta_l2h_count"]),
        "QWK": (float(dev["b0_QWK_rounded_mean"]), float(dev["b0_QWK_rounded_sd"]), float(dev["exp51_QWK_rounded_mean"]), float(dev["exp51_QWK_rounded_sd"]), float(dev["delta_QWK_rounded_mean"])),
    }

    main_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{测试集上 Hard-only 与 HMSA 的三种子结果（均值 $\pm$ 跨三个训练随机种子的样本标准差）。差值为 HMSA$-$Hard-only，并由未四舍五入的数值计算。}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"指标 & Hard-only & HMSA & $\Delta$ \\",
        r"\midrule",
    ]
    for metric in ("MAE", "Exact", "Kendall", "Bias", "L2H", "QWK"):
        row = test_rows[mapping[metric]]
        digits = 1 if metric == "L2H" else 3
        main_lines.append(
            f"{display_names[metric]} & {pm(float(row['b0_mean']), float(row['b0_sd']), digits)} & "
            f"\\textbf{{{pm(float(row['exp51_mean']), float(row['exp51_sd']), digits)}}} & "
            f"{signed(float(row['delta_mean']), digits)} \\\\"
        )
        if metric == "L2H":
            denominator = 103.0
            main_lines.append(
                f"L2H 比率 $\\downarrow$ & {pm(float(row['b0_mean']) / denominator, float(row['b0_sd']) / denominator)} & "
                f"\\textbf{{{pm(float(row['exp51_mean']) / denominator, float(row['exp51_sd']) / denominator)}}} & "
                f"{signed(float(row['delta_mean']) / denominator)} \\\\"
            )
    main_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write("tab_main_results", "\n".join(main_lines))

    dev_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{正式开发集上的三种子结果（均值 $\pm$ 样本标准差）。}",
        r"\label{tab:formal_dev_results}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"指标 & Hard-only & HMSA & $\Delta$ \\",
        r"\midrule",
    ]
    for metric in ("MAE", "Exact", "Kendall", "Bias", "L2H", "QWK"):
        b0, b0_sd, exp, exp_sd, delta = dev_values[metric]
        dev_lines.append(f"{display_names[metric]} & {pm(b0, b0_sd)} & \\textbf{{{pm(exp, exp_sd)}}} & {signed(delta)} \\\\")
    dev_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    # Do not emit a second unused "formal dev" table: the manuscript reports
    # the same development evidence in the matched control and appendix tables.


def recall_table() -> None:
    rows = read_csv(DATA / "test_recall_by_label.csv")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{测试集上的分等级召回率（均值 $\pm$ 跨三个训练随机种子的样本标准差）；$n$ 表示各等级样本数，差值由未四舍五入的数值计算。}",
        r"\label{tab:recall}",
        r"\small",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"评分等级 & $n$ & Hard-only & HMSA & $\Delta$ \\",
        r"\midrule",
    ]
    for row in rows:
        b0 = float(row["b0_mean"])
        exp51 = float(row["exp51_mean"])
        b0_text = pm(b0, float(row["b0_sd"]))
        exp51_text = pm(exp51, float(row["exp51_sd"]))
        if exp51 >= b0:
            exp51_text = f"\\textbf{{{exp51_text}}}"
        else:
            b0_text = f"\\textbf{{{b0_text}}}"
        n_text = f"{int(row['n']):,}"
        lines.append(f"{row['label']} & {n_text} & {b0_text} & {exp51_text} & {signed(float(row['delta']))} \\\\ ")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write("tab_recall", "\n".join(lines))


def diagnostic_seed42_table() -> None:
    rows = read_csv(DATA / "dev_seed42_diagnostics.csv")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{开发集随机种子 42 的诊断结果。Direct-soft 与 Single-head mix 仅用于预设推进判定；“未推进”表示未通过门槛，不代表相应方法族普遍无效。}",
        r"\label{tab:diagnostic_seed42}",
        r"\small",
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"配置 & 轮次 & MAE $\downarrow$ & 精确匹配率 $\uparrow$ & Kendall $\tau_b$ $\uparrow$ & 有符号偏差 & L2H & 结论 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & {row['epoch']} & {float(row['mae']):.3f} & {float(row['exact']):.3f} & "
            f"{float(row['kendall']):.3f} & {float(row['bias']):+.3f} & {row['l2h_count']} & "
            f"{ {'baseline': '基线', 'NO-GO': '未推进', 'PASS': '推进'}[row['gate']] } \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write("tab_diagnostic_seed42", "\n".join(lines))


def historical_evaluator_table() -> None:
    historical = read_csv(DATA / "prior_audit_evaluator_results.csv")
    test_rows = {row["metric"]: row for row in read_csv(DATA / "test_main_results.csv")}
    current = (
        (
            "Hard-only",
            test_rows["MAE_human_mean"],
            test_rows["Bias_human_mean"],
            test_rows["Exact_rounded"],
            test_rows["Kendall_human_mean"],
            test_rows["BinAgreement_paper_3way"],
            "b0",
        ),
        (
            "HMSA",
            test_rows["MAE_human_mean"],
            test_rows["Bias_human_mean"],
            test_rows["Exact_rounded"],
            test_rows["Kendall_human_mean"],
            test_rows["BinAgreement_paper_3way"],
            "exp51",
        ),
    )
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{同一教育评分任务上的既有结果与本文受控结果。A 组引自相关审计研究\cite{prioraudit}；本文的方法增益仅依据 B 组配对比较。}",
        r"\label{tab:historical_evaluators}",
        r"\small",
        r"\setlength{\tabcolsep}{3.6pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"评审器 & MAE $\downarrow$ & \shortstack{有符号偏差\\（0 最优）} & \shortstack{精确\\匹配率 $\uparrow$} & Kendall $\tau_b$ $\uparrow$ & \shortstack{三档\\一致率 $\uparrow$} \\",
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\textit{A. 相关工作报告的系统结果}} \\",
        r"\addlinespace[2pt]",
    ]
    for row in historical:
        bin_text = f"{float(row['bin_agreement']):.3f}"
        lines.append(
            f"{row['evaluator']} & {float(row['mae']):.3f} & {float(row['bias']):+.3f} & "
            f"{float(row['exact']):.3f} & {float(row['kendall']):.3f} & {bin_text} \\\\"
        )
    lines.extend([
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\textit{B. 本文受控比较（相同划分、配置和种子）}} \\",
        r"\addlinespace[2pt]",
    ])
    for evaluator, mae, bias, exact, kendall, bin_agreement, prefix in current:
        values = []
        for row in (mae, bias, exact, kendall, bin_agreement):
            values.append(pm(float(row[f"{prefix}_mean"]), float(row[f"{prefix}_sd"])))
        if evaluator == "HMSA":
            values = [f"\\textbf{{{value}}}" for value in values]
        lines.append(f"{evaluator} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write("tab_historical_evaluators", "\n".join(lines))


def disagreement_table() -> None:
    rows = {row["group"]: row for row in read_csv(DATA / "dev_disagreement_summary.csv")}
    labels = (("unanimous", "三人一致"), ("adjacent_2_to_1", "相邻等级2:1分歧"))
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{开发集上的评分一致性分层分析（均值 $\pm$ 跨三个训练随机种子的样本标准差）。}",
        r"\label{tab:dev_disagreement}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"分组 & 指标 & $n$ & Hard-only & HMSA & $\Delta$ \\",
        r"\midrule",
    ]
    metrics = (
        ("MAE", "MAE_human_mean"),
        ("精确匹配率", "Exact_rounded"),
        ("有符号偏差", "Bias_human_mean"),
        ("QWK", "QWK_rounded"),
    )
    for group_index, (group, group_label) in enumerate(labels):
        row = rows[group]
        for metric_index, (display, key) in enumerate(metrics):
            label = group_label if metric_index == 0 else ""
            n = row["n"] if metric_index == 0 else ""
            lines.append(
                f"{label} & {display} & {n} & "
                f"{pm(float(row[f'b0_{key}_mean']), float(row[f'b0_{key}_sd']))} & "
                f"\\textbf{{{pm(float(row[f'exp51_{key}_mean']), float(row[f'exp51_{key}_sd']))}}} & "
                f"{signed(float(row[f'delta_{key}_mean']))} \\\\"
            )
        if group_index == 0:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write("tab_dev_disagreement", "\n".join(lines))


def mechanism_control_tables() -> None:
    rows = {
        (row["method"], row["metric"]): row
        for row in read_csv(DATA / "dev_mechanism_control_summary.csv")
    }
    methods = ("Hard-only", "Shuffled-soft", "HMSA")
    metrics = (
        ("MAE_human_mean", r"MAE $\downarrow$", "lower"),
        ("Exact_rounded", r"精确匹配率 $\uparrow$", "higher"),
        ("Kendall_human_mean", r"Kendall $\tau_b$ $\uparrow$", "higher"),
        ("Bias_human_mean", r"有符号偏差（0 最佳）", "absolute"),
        ("QWK_rounded", r"QWK $\uparrow$", "higher"),
        ("L2H_count", r"L2H 数量 $\downarrow$", "lower"),
    )
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{事后开发集机制对照（均值 $\pm$ 跨三个训练随机种子的样本标准差）。Shuffled-soft 保留各硬标签内软目标的完整多重集，但打乱样本与软目标的对应关系；$\Delta_{\mathrm{H-S}}$ 为 HMSA$-$Shuffled-soft。}",
        r"\label{tab:dev_mechanism_summary}",
        r"\small",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"指标 & Hard-only & Shuffled-soft & HMSA & $\Delta_{\mathrm{H-S}}$ \\",
        r"\midrule",
    ]
    for key, display, direction in metrics:
        values = {method: float(rows[(method, key)]["mean"]) for method in methods}
        if direction == "lower":
            best = min(methods, key=lambda method: values[method])
        elif direction == "higher":
            best = max(methods, key=lambda method: values[method])
        else:
            best = min(methods, key=lambda method: abs(values[method]))
        cells = []
        digits = 1 if key == "L2H_count" else 3
        for method in methods:
            row = rows[(method, key)]
            cell = pm(float(row["mean"]), float(row["sample_sd"]), digits)
            cells.append(f"\\textbf{{{cell}}}" if method == best else cell)
        delta = values["HMSA"] - values["Shuffled-soft"]
        lines.append(f"{display} & " + " & ".join(cells) + f" & {signed(delta, digits)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write("tab_dev_mechanism_summary", "\n".join(lines))

    seed_rows = read_csv(DATA / "dev_mechanism_control_by_seed.csv")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{事后开发集机制对照的逐种子结果。轮次表示按相同开发集精确匹配率规则选出的检查点；三个主指标均只读取硬标签主头。}",
        r"\label{tab:dev_mechanism_by_seed}",
        r"\small",
        r"\begin{tabular}{clrrrr}",
        r"\toprule",
        r"种子 & 配置 & 轮次 & MAE $\downarrow$ & 精确匹配率 $\uparrow$ & Kendall $\tau_b$ $\uparrow$ \\",
        r"\midrule",
    ]
    for row_index, row in enumerate(seed_rows):
        seed = row["seed"] if row_index % 3 == 0 else ""
        method = row["method"]
        mae = f"{float(row['mae']):.3f}"
        exact = f"{float(row['exact']):.3f}"
        kendall = f"{float(row['kendall']):.3f}"
        if method == "HMSA":
            mae, exact, kendall = (f"\\textbf{{{value}}}" for value in (mae, exact, kendall))
        lines.append(
            f"{seed} & {method} & {row['selected_epoch']} & {mae} & {exact} & {kendall} \\\\"
        )
        if row_index in (2, 5):
            lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write("tab_dev_mechanism_by_seed", "\n".join(lines))


def meanaux_control_tables() -> None:
    summary = read_csv(DATA / "dev_meanaux_summary.csv")
    audit = read_json(DATA / "meanaux_audit_summary.json")
    if not (
        audit["valid"]
        and audit["checkpoint_paths_unique"]
        and audit["prediction_hashes_unique"]
        and audit["test_access_count"] == 0
        and audit["hmsa_better_than_meanaux_on_all_three_primary_metrics_for_every_seed"]
    ):
        raise ValueError("MeanAux audit contract is not satisfied")
    audit_names = {"Hard-only": "Hard-only", "MeanAux": "MeanAux", "HMSA": "HMSA"}
    for row in summary:
        frozen = audit["aggregate"][audit_names[row["method"]]]
        for csv_key, audit_key in (("mae", "MAE"), ("exact", "Exact"), ("kendall", "Kendall")):
            if abs(float(row[f"{csv_key}_mean"]) - float(frozen[audit_key][0])) > 1e-12:
                raise ValueError(f"MeanAux aggregate drift: {row['method']} {csv_key} mean")
            if abs(float(row[f"{csv_key}_sd"]) - float(frozen[audit_key][1])) > 1e-12:
                raise ValueError(f"MeanAux aggregate drift: {row['method']} {csv_key} sd")
    labels = {"Hard-only": "Hard-only", "MeanAux": "MeanAux", "HMSA": "HMSA"}
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{连续均分辅助目标的事后开发集匹配对照（均值 $\pm$ 跨三个训练随机种子的样本标准差）。MeanAux 与 HMSA 具有相同的双头结构和硬头评分路径，只将辅助目标及损失替换为连续人类均分与 Smooth L1。}",
        r"\label{tab:dev_meanaux_summary}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"配置 & MAE $\downarrow$ & 精确匹配率 $\uparrow$ & Kendall $\tau_b$ $\uparrow$ \\",
        r"\midrule",
    ]
    for row in summary:
        cells = (
            pm(float(row["mae_mean"]), float(row["mae_sd"])),
            pm(float(row["exact_mean"]), float(row["exact_sd"])),
            pm(float(row["kendall_mean"]), float(row["kendall_sd"])),
        )
        if row["method"] == "HMSA":
            cells = tuple(f"\\textbf{{{cell}}}" for cell in cells)
        lines.append(f"{labels[row['method']]} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write("tab_dev_meanaux_summary", "\n".join(lines))

    seed_rows = read_csv(DATA / "dev_meanaux_by_seed.csv")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{连续均分辅助目标对照的逐种子开发集结果。轮次由硬头开发集精确匹配率选择；所有指标均只读取硬头。}",
        r"\label{tab:dev_meanaux_by_seed}",
        r"\small",
        r"\begin{tabular}{clrrrr}",
        r"\toprule",
        r"种子 & 配置 & 轮次 & MAE $\downarrow$ & 精确匹配率 $\uparrow$ & Kendall $\tau_b$ $\uparrow$ \\",
        r"\midrule",
    ]
    for row_index, row in enumerate(seed_rows):
        seed = row["seed"] if row_index % 3 == 0 else ""
        values = (
            f"{float(row['mae']):.3f}",
            f"{float(row['exact']):.3f}",
            f"{float(row['kendall']):.3f}",
        )
        if row["method"] == "HMSA":
            values = tuple(f"\\textbf{{{value}}}" for value in values)
        lines.append(
            f"{seed} & {row['method']} & {row['selected_epoch']} & "
            + " & ".join(values)
            + r" \\"
        )
        if row_index in (2, 5):
            lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write("tab_dev_meanaux_by_seed", "\n".join(lines))


def main() -> None:
    methods_table()
    positioning_table()
    protocol_table()
    main_results_table_v2()
    recall_table()
    diagnostic_seed42_table()
    historical_evaluator_table()
    disagreement_table()
    mechanism_control_tables()
    meanaux_control_tables()
    print(f"Generated tables in {TABLES}")


if __name__ == "__main__":
    main()
