"""ReportType / ReportSource 枚举.

原来的实现只有 ReportType.Web/Local 两个值(实际对应原版的 ReportSource)。
现改为薄 re-export: 完整枚举在 gpt_researcher/utils/enum.py(与原版一致)。
"""

from .utils.enum import ReportType, ReportSource

__all__ = ["ReportType", "ReportSource"]
