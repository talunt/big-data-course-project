"""Summaries for a single self-contained reproduction run."""

import json
from .common import MODEL_NAMES, read_csv


def write_run_summary(result):
    extended = result / "extended"
    review = json.loads((result / "review/review.json").read_text(encoding="utf-8"))
    selected = review["original_validation_selection"]
    recommended = review["recommended_family"]
    lines = [
        "# 复现结果",
        "",
        "数据准备、训练、模型恢复及独立核查均已完成。",
        "",
        f"单次验证集选择：{MODEL_NAMES[selected]}。",
        f"综合历史比较推荐：{MODEL_NAMES[recommended]}。",
        "",
        "| 方法 | 测试RMSE | 测试MAE | 测试R² |",
        "|---|---:|---:|---:|",
    ]
    for row in read_csv(extended / "metrics.csv"):
        lines.append(
            f"|{MODEL_NAMES[row['family']]}|{float(row['rmse']):.2f}|{float(row['mae']):.2f}|{float(row['r2']):.3f}|"
        )
    lines += [
        "",
        "综合推荐依据两个早期窗口的合并RMSE，并结合测试结果解释。这是已观察历史结果的回顾性比较，后续用新增日期验证。",
        "",
        "- 综合比较与推荐模型的误差分析：[review/summary.md](review/summary.md)",
        "- 推荐模型图表：review/figures/",
        "- 原始对照与扩展实验：baseline/、extended/",
        "- 重新训练的模型：storage/（采用HDFS时以运行记录中的位置为准）",
        "- 运行记录与日志：logs/、manifest.json",
    ]
    (result / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
