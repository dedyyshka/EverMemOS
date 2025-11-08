from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from memory_layer.types import Memory
from agentic_layer.memory_models import (
    MemoryType,
    Metadata,
    MemoryModel,
    RetrieveMethod,
)


@dataclass
class FetchMemRequest:
    """记忆获取请求"""

    user_id: str
    limit: Optional[int] = 40
    offset: Optional[int] = 0
    filters: Optional[Dict[str, Any]] = field(default_factory=dict)
    memory_type: Optional[MemoryType] = MemoryType.MULTIPLE
    sort_by: Optional[str] = None
    sort_order: str = "desc"  # "asc" or "desc"
    version_range: Optional[tuple[Optional[str], Optional[str]]] = (
        None  # 版本范围 (start, end)，左闭右闭区间 [start, end]
    )

    def get_memory_types(self) -> List[MemoryType]:
        """获取要查询的记忆类型列表"""
        if self.memory_type == MemoryType.MULTIPLE:
            # 当为MULTIPLE时，返回BASE_MEMORY、PROFILE、PREFERENCE三类
            return [MemoryType.BASE_MEMORY, MemoryType.PROFILE, MemoryType.PREFERENCE]
        else:
            return [self.memory_type]


@dataclass
class FetchMemResponse:
    """记忆获取响应"""

    memories: List[MemoryModel]
    total_count: int
    has_more: bool = False
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class RetrieveMemRequest:
    """记忆检索请求"""

    user_id: str
    memory_types: List[MemoryType] = field(default_factory=list)
    top_k: int = 40
    filters: Dict[str, Any] = field(default_factory=dict)
    include_metadata: bool = True
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    query: Optional[str] = None  # retrieve的时候
    retrieve_method: RetrieveMethod = field(default=RetrieveMethod.KEYWORD)
    memory_sub_type: Optional[str] = None  # 记忆子类型过滤：episode/semantic_memory/event_log
    semantic_start_time: Optional[str] = None  # 语义记忆开始时间过滤
    semantic_end_time: Optional[str] = None  # 语义记忆结束时间过滤


@dataclass
class RetrieveMemResponse:
    """记忆检索响应"""

    memories: List[Dict[str, List[Memory]]]
    scores: List[Dict[str, List[float]]]
    importance_scores: List[float] = field(default_factory=list)  # 新增：群组重要性得分
    original_data: List[Dict[str, List[Dict[str, Any]]]] = field(
        default_factory=list
    )  # 新增：原始数据
    total_count: int = 0
    has_more: bool = False
    query_metadata: Metadata = field(default_factory=Metadata)
    metadata: Metadata = field(default_factory=Metadata)
