"""
AskUserQuestion 工具 —— Agent 向用户确认/澄清需求
"""

from src.tools.base import BaseTool, ToolResult


class AskUserQuestionTool(BaseTool):
    """向用户提问以澄清需求或获取确认"""

    name = "ask_user_question"
    description = (
        "向用户提出问题以澄清需求、获取确认或提供选项。"
        "适用于：(1) 任务描述模糊需要澄清 (2) 有多种实现方案需要选择 "
        "(3) 高风险操作前需要用户确认 (4) 需要用户提供额外信息才能继续。"
        "支持单选和多选两种问题类型。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "description": "要问的问题列表 (1-4 个)",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "完整的问题文本",
                        },
                        "header": {
                            "type": "string",
                            "description": "简短标签（最多12字符）",
                        },
                        "options": {
                            "type": "array",
                            "description": "可选项 (2-4个)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "description": "选项显示文本",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "选项说明",
                                    },
                                },
                                "required": ["label", "description"],
                            },
                        },
                        "multiSelect": {
                            "type": "boolean",
                            "description": "是否允许多选",
                            "default": False,
                        },
                    },
                    "required": ["question", "header", "options"],
                },
            },
        },
        "required": ["questions"],
    }

    def execute(self, questions: list[dict]) -> ToolResult:
        if not questions:
            return ToolResult(
                success=False,
                content="问题列表不能为空",
            )

        print("\n" + "=" * 60)
        print("[Agent 需要确认]")
        print("=" * 60)

        answers = {}
        for idx, q in enumerate(questions, 1):
            question = q.get("question", "")
            header = q.get("header", f"Q{idx}")
            options = q.get("options", [])
            multi = q.get("multiSelect", False)

            print(f"\n--- {header} ---")
            print(f"{question}\n")

            for i, opt in enumerate(options, 1):
                print(f"  [{i}] {opt.get('label', '')}")
                desc = opt.get("description", "")
                if desc:
                    print(f"      {desc}")

            if multi:
                print(f"\n  输入选项编号（多选用逗号分隔，如 1,3）:")
            else:
                print(f"\n  输入选项编号 (1-{len(options)}):")

            try:
                raw = input("  > ").strip()
                if not raw:
                    answers[header] = []
                    continue

                if multi:
                    indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
                    answers[header] = [
                        options[i-1].get("label", "")
                        for i in indices if 1 <= i <= len(options)
                    ]
                else:
                    idx_opt = int(raw)
                    if 1 <= idx_opt <= len(options):
                        answers[header] = options[idx_opt - 1].get("label", "")
                    else:
                        answers[header] = "无效选择"
            except (ValueError, EOFError):
                answers[header] = "未回答"

        print("\n" + "-" * 40)

        # 格式化结果给 LLM
        result_lines = ["[用户回答]"]
        for header, answer in answers.items():
            if isinstance(answer, list):
                result_lines.append(f"- {header}: {', '.join(answer) if answer else '未选择'}")
            else:
                result_lines.append(f"- {header}: {answer}")

        return ToolResult(
            success=True,
            content="\n".join(result_lines),
            metadata={"answers": answers},
        )
