"""
语义记忆 Milvus 转换器

负责将 MongoDB 的语义记忆文档转换为 Milvus Collection 实体，支持个人与群组。
"""

from typing import Dict, Any
import json
from datetime import datetime

from core.oxm.milvus.base_converter import BaseMilvusConverter
from core.observation.logger import get_logger
from infra_layer.adapters.out.search.milvus.memory.semantic_memory_collection import (
    SemanticMemoryCollection,
)
from infra_layer.adapters.out.persistence.document.memory.semantic_memory_record import (
    SemanticMemoryRecord as MongoSemanticMemoryRecord,
)

logger = get_logger(__name__)


class SemanticMemoryMilvusConverter(BaseMilvusConverter[SemanticMemoryCollection]):
    """
    语义记忆 Milvus 转换器
    
    将 MongoDB 的语义记忆文档转换为 Milvus Collection 实体。
    使用独立的 SemanticMemoryCollection，支持个人和群组语义记忆。
    """

    @classmethod
    def _parse_time_field(cls, time_value, field_name: str, doc_id) -> int:
        """解析时间字段，失败时返回 0 并记录警告"""
        if not time_value:
            return 0
        
        try:
            if isinstance(time_value, datetime):
                return int(time_value.timestamp())
            elif isinstance(time_value, str):
                dt = datetime.fromisoformat(time_value.replace('Z', '+00:00'))
                return int(dt.timestamp())
            elif isinstance(time_value, (int, float)):
                return int(time_value)
        except Exception as e:
            logger.warning(f"解析 {field_name} 失败 (doc_id={doc_id}): {time_value}, 错误: {e}")
        
        return 0

    @classmethod
    def from_mongo(cls, source_doc: MongoSemanticMemoryRecord) -> Dict[str, Any]:
        """
        从 MongoDB 语义记忆文档转换为 Milvus Collection 实体

        Args:
            source_doc: MongoDB 语义记忆文档实例

        Returns:
            Dict[str, Any]: Milvus 实体字典，可直接用于插入
        """
        if source_doc is None:
            raise ValueError("MongoDB 文档不能为空")

        try:
            # 解析时间字段
            start_time = cls._parse_time_field(source_doc.start_time, "start_time", source_doc.id)
            end_time = cls._parse_time_field(source_doc.end_time, "end_time", source_doc.id)
            
            # 构建搜索内容
            search_content = cls._build_search_content(source_doc)
            
        
            
            # 创建 Milvus 实体字典
            milvus_entity = {
                # 基础标识字段
                "id": str(source_doc.id),  # 使用 Beanie 的 id 属性
                "user_id": source_doc.user_id or "",
                "group_id": source_doc.group_id or "",
                "participants": source_doc.participants if source_doc.participants else [],
                "parent_episode_id": source_doc.parent_episode_id or "",
                # 时间字段
                "start_time": start_time,
                "end_time": end_time,
                "duration_days": source_doc.duration_days if source_doc.duration_days else 0,
                # 核心内容字段
                "content": source_doc.content,
                "evidence": source_doc.evidence or "",
                "search_content": search_content,
                # 详细信息 JSON
                "metadata": json.dumps(cls._build_detail(source_doc), ensure_ascii=False),
                # 审计字段
                "created_at": (
                    int(source_doc.created_at.timestamp())
                    if source_doc.created_at
                    else 0
                ),
                "updated_at": (
                    int(source_doc.updated_at.timestamp())
                    if source_doc.updated_at
                    else 0
                ),
                # 向量字段
                "vector": source_doc.vector if source_doc.vector else [],
            }

            return milvus_entity

        except Exception as e:
            logger.error("从 MongoDB 语义记忆文档转换为 Milvus 实体失败: %s", e)
            raise

    @classmethod
    def _build_detail(cls, source_doc: MongoSemanticMemoryRecord) -> Dict[str, Any]:
        """构建详细信息字典"""
        detail = {
            "vector_model": source_doc.vector_model,
            "extend": source_doc.extend,
        }
        
        # 过滤掉 None 值
        return {k: v for k, v in detail.items() if v is not None}

    @staticmethod
    def _build_search_content(source_doc: MongoSemanticMemoryRecord) -> str:
        """构建搜索内容（JSON 列表格式）"""
        text_content = []
        
        # 主要内容
        if source_doc.content:
            text_content.append(source_doc.content)
        
        # 添加证据信息以提升检索能力
        if source_doc.evidence:
            text_content.append(source_doc.evidence)
        
        return json.dumps(text_content, ensure_ascii=False)

