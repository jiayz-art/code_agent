"""
TodoWrite 任务管理工具 —— Agent 自我追踪任务进度
"""

from src.tools.base import BaseTool, ToolResult


class TodoWriteTool(BaseTool):
    """任务清单管理工具 —— Agent 用此工具追踪复杂任务的进度"""

    name = "todo_write"
    description = (
        "管理和追踪复杂任务的进度。当任务需要 3 个以上独立步骤时使用。"
        "每次完成一个步骤后更新对应 todo 的状态。"
        "状态: pending(未开始) / in_progress(进行中) / completed(已完成)。"
        "同一时间只应有一个 in_progress 的任务。"
        "任务完成后不需要手动标记所有为 completed — 工具会记录最终状态。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "任务列表（覆盖式更新，传入当前全部任务）",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "任务描述（祈使句，如「修复登录 bug」）",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "任务状态",
                        },
                        "activeForm": {
                            "type": "string",
                            "description": "进行时描述（如「正在修复登录 bug」）",
                        },
                    },
                    "required": ["content", "status", "activeForm"],
                },
            },
        },
        "required": ["todos"],
    }

    def execute(self, todos: list[dict]) -> ToolResult:
        if not todos:
            return ToolResult(
                success=False,
                content="todo 列表不能为空",
            )

        # 统计
        total = len(todos)
        pending = sum(1 for t in todos if t.get("status") == "pending")
        in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
        completed = sum(1 for t in todos if t.get("status") == "completed")

        # 格式化输出
        lines = [f"[任务追踪] 进度: {completed}/{total} 已完成"]
        for i, todo in enumerate(todos, 1):
            status_icon = {"pending": "  ", "in_progress": "▶ ", "completed": "✓ "}
            icon = status_icon.get(todo.get("status", "pending"), "? ")
            content = todo.get("content", "?")
            lines.append(f"  {i}. {icon}{content}")

        if in_progress > 0:
            in_progress_items = [t.get("activeForm", t.get("content", ""))
                                for t in todos if t.get("status") == "in_progress"]
            lines.append(f"\n当前进行中: {', '.join(in_progress_items)}")

        return ToolResult(
            success=True,
            content="\n".join(lines),
            metadata={
                "total": total,
                "pending": pending,
                "in_progress": in_progress,
                "completed": completed,
            },
        )
