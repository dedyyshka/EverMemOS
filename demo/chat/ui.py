"""终端 UI 工具

提供美观的终端输出格式。
"""

import re
from typing import List, Dict, Any, Optional

from demo.ui import I18nTexts
from common_utils.cli_ui import CLIUI


def extract_event_time_from_memory(mem: Dict[str, Any]) -> Optional[str]:
    """从记忆数据中提取事件实际发生时间
    
    提取优先级：
    1. subject 字段中的日期（括号格式，如 "(2025-08-26)"）
    2. subject 字段中的日期（中文格式，如 "2025年8月26日"）
    3. episode 内容中的日期（中文或 ISO 格式）
    4. timestamp / created_at / event_time 等时间字段
    5. 如果都提取不到，返回 None
    
    Args:
        mem: 记忆字典，包含 subject, episode, timestamp 等字段
        
    Returns:
        日期字符串，格式为 YYYY-MM-DD，或 None（无法提取）
        
    Examples:
        >>> mem = {"subject": "北京旅游建议 (2025-08-26)"}
        >>> extract_event_time_from_memory(mem)
        '2025-08-26'
        
        >>> mem = {"timestamp": "2025-08-26T10:30:00"}
        >>> extract_event_time_from_memory(mem)
        '2025-08-26'
        
        >>> mem = {"subject": "", "episode": ""}
        >>> extract_event_time_from_memory(mem)
        None
    """
    subject = mem.get("subject", "")
    episode = mem.get("episode", "")
    
    # 1. 从 subject 提取：匹配括号内的 ISO 日期格式 (YYYY-MM-DD)
    if subject:
        match = re.search(r'\((\d{4}-\d{2}-\d{2})\)', subject)
        if match:
            return match.group(1)
        
        # 2. 从 subject 提取：匹配中文日期格式 "YYYY年MM月DD日"
        match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', subject)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # 3. 从 episode 提取（搜索整个内容，不限制字符数）
    if episode:
        # 匹配 "于YYYY年MM月DD日" 或 "在YYYY年MM月DD日"
        match = re.search(r'[于在](\d{4})年(\d{1,2})月(\d{1,2})日', episode)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # 匹配 ISO 格式 "YYYY-MM-DD"
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', episode)
        if match:
            return match.group(0)
        
        # 匹配其他中文日期格式（不带"于/在"前缀）
        match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', episode)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # 4. 从时间字段提取（timestamp, created_at, event_time, updated_at）
    for time_field in ["timestamp", "event_time", "created_at", "updated_at"]:
        time_value = mem.get(time_field, "")
        if time_value:
            # 支持 ISO 格式 "YYYY-MM-DDTHH:MM:SS" 或 "YYYY-MM-DD HH:MM:SS"
            match = re.search(r'(\d{4}-\d{2}-\d{2})', str(time_value))
            if match:
                return match.group(1)
    
    # 5. 无法提取事件时间，返回 None
    return None


class ChatUI:
    """终端界面工具类"""
    
    @staticmethod
    def _ui() -> CLIUI:
        """获取 UI 实例"""
        return CLIUI()
    
    @staticmethod
    def clear_screen():
        """清空屏幕"""
        print("\033[2J\033[H", end="")
        import sys
        sys.stdout.flush()
    
    @staticmethod
    def print_banner(texts: I18nTexts):
        """打印欢迎横幅"""
        ui = ChatUI._ui()
        print()
        ui.banner(texts.get("banner_title"), subtitle=texts.get("banner_subtitle"))
        print()
    
    @staticmethod
    def print_group_list(groups: List[Dict[str, Any]], texts: I18nTexts):
        """显示群组列表"""
        ui = ChatUI._ui()
        print()
        ui.section_heading(texts.get("groups_available_title"))
        
        rows = []
        for group in groups:
            index = group["index"]
            group_id = group["group_id"]
            name = group.get("name", group_id)
            count = group["memcell_count"]
            count_text = f"💾 {count} " + ("memories" if texts.language == "en" else "条记忆")
            rows.append([f"[{index}]", group_id, f'📝 "{name}"', count_text])
        
        headers = [
            texts.get("table_header_index"),
            texts.get("table_header_group"),
            texts.get("table_header_name"),
            texts.get("table_header_count"),
        ]
        ui.table(headers=headers, rows=rows, aligns=["right", "left", "left", "right"])
    
    @staticmethod
    def print_retrieved_memories(
        memories: List[Dict[str, Any]],
        texts: I18nTexts,
        retrieval_metadata: Optional[Dict[str, Any]] = None,
    ):
        """显示检索到的记忆"""
        ui = ChatUI._ui()
        
        heading = f"🔍 {texts.get('retrieval_complete')}"
        shown_count = len(memories)
        if shown_count > 0:
            heading += f" - {texts.get('retrieval_showing', shown=shown_count)}"
        
        # 显示检索模式和耗时
        if retrieval_metadata:
            retrieval_mode = retrieval_metadata.get("retrieval_mode", "rrf")
            latency_ms = retrieval_metadata.get("total_latency_ms", 0.0)
            
            # 国际化检索模式显示
            mode_map = {
                "rrf": texts.get("agentic_mode_rrf"),
                "embedding": texts.get("agentic_mode_embedding"),
                "bm25": texts.get("agentic_mode_bm25"),
                "agentic": texts.get("agentic_mode_agentic"),
                "agentic_fallback": texts.get("agentic_mode_agentic_fallback"),
            }
            mode_text = mode_map.get(retrieval_mode, retrieval_mode)
            heading += f" | {mode_text} | {int(latency_ms)}ms"
        
        ui.section_heading(heading)
        
        # 🔥 Agentic 检索特殊信息显示
        if retrieval_metadata and retrieval_metadata.get("retrieval_mode") == "agentic":
            agentic_info = []
            
            # LLM 判断结果（国际化）
            is_sufficient = retrieval_metadata.get("is_sufficient")
            if is_sufficient is not None:
                status_icon = "✅" if is_sufficient else "❌"
                status_text = texts.get("agentic_sufficient") if is_sufficient else texts.get("agentic_insufficient")
                agentic_info.append(f"{texts.get('agentic_llm_judgment')}: {status_icon} {status_text}")
            
            # 是否多轮（国际化）
            is_multi_round = retrieval_metadata.get("is_multi_round", False)
            if is_multi_round:
                agentic_info.append(f"🔄 {texts.get('agentic_multi_round')}")
                
                # 改进查询
                refined_queries = retrieval_metadata.get("refined_queries", [])
                if refined_queries:
                    agentic_info.append(f"{texts.get('agentic_generated_queries')}: {len(refined_queries)}")
            else:
                agentic_info.append(f"⚡ {texts.get('agentic_single_round')}")
            
            # Round 统计（国际化）
            round1_count = retrieval_metadata.get("round1_count", 0)
            round2_count = retrieval_metadata.get("round2_count", 0)
            items_text = texts.get("agentic_items")
            if round1_count:
                agentic_info.append(f"{texts.get('agentic_round1_count')}: {round1_count} {items_text}")
            if round2_count:
                agentic_info.append(f"{texts.get('agentic_round2_count')}: {round2_count} {items_text}")
            
            if agentic_info:
                print()
                ui.note(" | ".join(agentic_info), icon="🤖")
                
                # 显示 LLM 推理（国际化优化提示语）
                reasoning = retrieval_metadata.get("reasoning")
                if reasoning:
                    # 优化常见的误导性提示（国际化）
                    # 检测中文内容并替换为国际化文本
                    chinese_keywords = [
                        "为空", "均为空", "内容为空", "记忆内容",
                        "未提供", "不足", "无法提供", "相关性",
                        "检索到的记忆", "信息不够"
                    ]
                    if any(kw in reasoning for kw in chinese_keywords):
                        reasoning = texts.get("agentic_reasoning_hint")
                    
                    print(f"   💭 {reasoning}")
                
                # 显示改进查询（国际化）
                if is_multi_round:
                    refined_queries = retrieval_metadata.get("refined_queries", [])
                    if refined_queries:
                        print(f"   🔍 {texts.get('agentic_supplementary_queries')} ({len(refined_queries)}):")
                        for i, q in enumerate(refined_queries[:3], 1):
                            print(f"      {i}. {q[:60]}{'...' if len(q) > 60 else ''}")
        
        # 显示记忆列表
        lines = []
        for i, mem in enumerate(memories, start=1):
            # 提取事件实际发生时间（不是存储时间）
            event_time = extract_event_time_from_memory(mem)
            
            # 优先级：subject > summary > episode > atomic_fact > content
            # 使用 strip() 确保空字符串被正确处理
            subject = (mem.get("subject") or "").strip()
            summary = (mem.get("summary") or "").strip()
            episode = (mem.get("episode") or "").strip()
            atomic_fact = (mem.get("atomic_fact") or "").strip()
            content = (mem.get("content") or "").strip()
            
            # 选择第一个非空的字段
            display_text = subject or summary or episode or atomic_fact or content or "(无内容)"
            
            # 限制显示长度
            if len(display_text) > 80:
                display_text = display_text[:77] + "..."
            
            # 构建显示行：有时间则显示时间，无时间则省略
            if event_time:
                lines.append(f"📌 [{i}]  {event_time}  │  {display_text}")
            else:
                lines.append(f"📌 [{i}]  {display_text}")
        
        if lines:
            print()
            ui.panel(lines)
    
    @staticmethod
    def print_generating_indicator(texts: I18nTexts):
        """显示生成进度提示"""
        ui = ChatUI._ui()
        print()
        ui.note(f"🤔 {texts.get('chat_generating')}", icon="⏳")
    
    @staticmethod
    def print_generation_complete(texts: I18nTexts):
        """清除生成提示并显示完成标识"""
        print("\r\033[K", end="")
        print("\033[A\033[K", end="")
        print("\033[A\033[K", end="")
        ui = ChatUI._ui()
        ui.success(f"✓ {texts.get('chat_generation_complete')}")
    
    @staticmethod
    def clear_progress_indicator():
        """清除进度提示"""
        print("\r\033[K", end="")
        print("\033[A\033[K", end="")
        print("\033[A\033[K", end="")
    
    @staticmethod
    def print_assistant_response(response: str, texts: I18nTexts):
        """显示助手回答
        
        优化后的显示：
        - 主要显示 answer（大标题）
        - references 和 confidence 作为元数据（小字）
        - 隐藏 reasoning
        """
        ui = ChatUI._ui()
        print()
        
        # 尝试解析 JSON 响应
        try:
            import json
            data = json.loads(response)
            
            # 提取字段
            answer = data.get("answer", "")
            references = data.get("references", [])
            confidence = data.get("confidence", "")
            
            # 显示主回答（大标题）
            ui.panel([answer], title=f"🤖 {texts.get('response_assistant_title')}")
            
            # 显示元数据（小字，弱化显示）
            metadata_parts = []
            if references:
                ref_text = ", ".join(references)
                metadata_parts.append(f"📚 {ref_text}")
            if confidence:
                confidence_icon = {"high": "✓", "medium": "~", "low": "?"}.get(confidence, "")
                metadata_parts.append(f"{confidence_icon} {confidence}")
            
            if metadata_parts:
                metadata_line = "  │  ".join(metadata_parts)
                print(f"  {metadata_line}")
            
        except (json.JSONDecodeError, ValueError):
            # 如果不是 JSON 格式，直接显示原始回答
            ui.panel([response], title=f"🤖 {texts.get('response_assistant_title')}")
        
        ui.rule()
        print()
    
    @staticmethod
    def print_help(texts: I18nTexts):
        """显示帮助信息"""
        ui = ChatUI._ui()
        print()
        ui.section_heading(texts.get("cmd_help_title"))
        lines = [
            f"🚪  {texts.get('cmd_exit')}",
            f"🧹  {texts.get('cmd_clear')}",
            f"🔄  {texts.get('cmd_reload')}",
            f"❓  {texts.get('cmd_help')}",
        ]
        ui.panel(lines)
        print()
    
    @staticmethod
    def print_info(message: str, texts: I18nTexts):
        """显示信息提示"""
        ui = ChatUI._ui()
        print()
        ui.success(f"✓ {message}")
        print()
    
    @staticmethod
    def print_error(message: str, texts: I18nTexts):
        """显示错误信息"""
        ui = ChatUI._ui()
        print()
        ui.error(f"✗ {message}")
        print()

