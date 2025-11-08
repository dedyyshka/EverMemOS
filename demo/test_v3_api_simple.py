"""V3 API 简单测试 - 实现与 extract_memory 同款功能

功能：
1. 清空数据库（MongoDB、ES、Milvus）
2. 使用 V3 API 方式提取记忆
3. 包含：MemCell + 语义记忆 + Event Log
4. 同步到 Milvus/ES
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

from common_utils.datetime_utils import from_iso_format, get_now_with_timezone
from memory_layer.types import RawDataType
from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from memory_layer.memory_manager import MemorizeRequest
from agentic_layer.memory_manager import MemoryManager
from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)


DATA_FILE = Path("/Users/admin/Documents/Projects/opensource/memsys-opensource/data/assistant_chat_zh.json")


async def clear_all_data():
    """清空 MongoDB、ES、Milvus 的数据"""
    print("=" * 80)
    print("清空数据库")
    print("=" * 80)
    
    # 1. 清空 MongoDB MemCell
    print("\n[MongoDB] 清空 MemCell 集合...")
    memcell_repo = get_bean_by_type(MemCellRawRepository)
    from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell
    deleted_count = await MemCell.delete_all()
    print(f"  ✅ 已删除 {deleted_count.deleted_count if deleted_count else 0} 条 MemCell")
    
    # 2. 清空 Milvus（删除所有记录）
    print("\n[Milvus] 清空记忆集合...")
    milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    # 使用 collection.delete() 删除所有记录（id != "" 永远为真）
    delete_result = await milvus_repo.collection.delete(expr='id != ""')
    print(f"  ✅ 已删除 {delete_result.delete_count} 条记录")
    await milvus_repo.flush()
    print(f"  ✅ Milvus 已刷新")
    
    # 3. 清空 ES
    print("\n[ES] 清空索引...")
    es_repo = get_bean_by_type(EpisodicMemoryEsRepository)
    client = await es_repo.get_client()
    index_name = es_repo.get_index_name()
    
    # 删除所有文档
    delete_response = await client.delete_by_query(
        index=index_name,
        body={"query": {"match_all": {}}}
    )
    print(f"  ✅ 已删除 {delete_response.get('deleted', 0)} 条文档")
    
    # 刷新索引
    await client.indices.refresh(index=index_name)
    print(f"  ✅ 索引已刷新")
    
    print("\n✅ 所有数据已清空")


def load_chat_data(file_path: Path) -> List[Dict[str, Any]]:
    """加载聊天数据"""
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        return data.get("conversation_list", [])
    return data


def normalize_message(entry: Dict[str, Any]) -> Dict[str, Any] | None:
    """归一化消息格式"""
    timestamp = (
        entry.get("create_time")
        or entry.get("createTime")
        or entry.get("timestamp")
        or entry.get("created_at")
    )
    if not timestamp:
        return None
    
    if isinstance(timestamp, str):
        timestamp_dt = from_iso_format(timestamp)
    else:
        return None
    
    speaker_name = entry.get("sender_name") or entry.get("sender")
    if not speaker_name:
        origin = entry.get("origin")
        if isinstance(origin, dict):
            speaker_name = origin.get("fullName") or origin.get("full_name")
    if not speaker_name:
        return None
    
    speaker_id = None
    origin = entry.get("origin")
    if isinstance(origin, dict):
        speaker_id = origin.get("createBy") or origin.get("create_by")
    if not speaker_id:
        speaker_id = entry.get("sender_id") or entry.get("sender")
    
    content = str(entry.get("content", ""))
    
    return {
        "speaker_id": str(speaker_id or speaker_name),
        "speaker_name": str(speaker_name),
        "content": content,
        "timestamp": timestamp_dt,
    }


async def extract_memories():
    """使用 V3 API 方式提取记忆"""
    print("\n" + "=" * 80)
    print("使用 V3 API 提取记忆")
    print("=" * 80)
    
    # 加载数据
    print(f"\n✓ 加载聊天数据: {DATA_FILE.name}")
    events = load_chat_data(DATA_FILE)
    print(f"  - 共 {len(events)} 条消息")
    
    # 初始化 MemoryManager
    manager = MemoryManager()
    
    # 逐条处理消息
    history: List[RawData] = []
    saved_count = 0
    
    print("\n✓ 开始处理消息...")
    
    for idx, entry in enumerate(events):
        # 归一化
        message_payload = normalize_message(entry)
        if not message_payload:
            continue
        
        message_id = (
            entry.get("message_id")
            or entry.get("id")
            or entry.get("uuid")
            or f"msg_{idx}"
        )
        
        raw_item = RawData(
            content=message_payload,
            data_id=str(message_id),
            data_type=RawDataType.CONVERSATION,
        )
        
        # 初始化
        if not history:
            history.append(raw_item)
            continue
        
        # 构建请求
        request = MemorizeRequest(
            history_raw_data_list=list(history),
            new_raw_data_list=[raw_item],
            raw_data_type=RawDataType.CONVERSATION,
            user_id_list=["default"],
            group_id="assistant",
            enable_semantic_extraction=True,   # ✅ 启用语义记忆
            enable_event_log_extraction=True,  # ✅ 启用 Event Log
        )
        
        # 调用 memorize
        result = await manager.memorize(request)
        
        if result:
            saved_count += 1
            print(f"  [{saved_count}] ✅ 提取成功，返回 {len(result)} 个 Memory")
            
            # 重置历史
            history = [raw_item]
        else:
            # 继续累积
            history.append(raw_item)
            if len(history) > 20:
                history = history[-20:]
    
    print(f"\n✅ 处理完成，共提取 {saved_count} 个 MemCell")
    return saved_count


async def verify_results(expected_count: int):
    """验证存储结果"""
    print("\n" + "=" * 80)
    print("验证存储结果")
    print("=" * 80)
    
    # 1. 验证 MongoDB
    print("\n[MongoDB] 检查 MemCell")
    memcell_repo = get_bean_by_type(MemCellRawRepository)
    memcells = await memcell_repo.find_by_group_id("assistant", limit=100)
    print(f"  - 找到 {len(memcells)} 个 MemCell")
    
    if memcells:
        # 统计内容
        total_semantic = 0
        total_eventlog = 0
        
        for memcell in memcells:
            if hasattr(memcell, 'semantic_memories') and memcell.semantic_memories:
                total_semantic += len(memcell.semantic_memories)
            if hasattr(memcell, 'event_log') and memcell.event_log:
                event_log = memcell.event_log
                if isinstance(event_log, dict):
                    total_eventlog += len(event_log.get('atomic_fact', []))
                else:
                    total_eventlog += len(getattr(event_log, 'atomic_fact', []))
        
        print(f"  - episode: {len(memcells)} 个")
        print(f"  - semantic_memories: {total_semantic} 个")
        print(f"  - event_log atomic_facts: {total_eventlog} 个")
        print(f"  - 预期 Milvus/ES 记录数: {len(memcells) + total_semantic + total_eventlog}")
    
    # 2. 验证 Milvus
    print("\n[Milvus] 检查记录")
    milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    
    # 查询所有记录（用一个通用向量）
    from agentic_layer.vectorize_service import get_vectorize_service
    vectorize_service = get_vectorize_service()
    query_vector = await vectorize_service.get_embedding("测试")
    
    milvus_results = await milvus_repo.vector_search(
        query_vector=query_vector,
        user_id="default",
        limit=1000,
    )
    print(f"  - 找到 {len(milvus_results)} 条记录")
    
    # 统计类型分布
    from collections import Counter
    if milvus_results:
        types = [r.get('memory_sub_type', 'unknown') for r in milvus_results]
        type_counts = Counter(types)
        print(f"  - 类型分布: {dict(type_counts)}")
    
    # 3. 验证 ES
    print("\n[ES] 检查记录")
    es_repo = get_bean_by_type(EpisodicMemoryEsRepository)
    
    # 查询所有记录
    es_results = await es_repo.multi_search(
        query=[],
        user_id="default",
        size=1000,
    )
    print(f"  - 找到 {len(es_results)} 条记录")
    
    # 统计类型分布
    if es_results:
        types = [r.get('_source', {}).get('type', 'unknown') for r in es_results]
        type_counts = Counter(types)
        print(f"  - 类型分布: {dict(type_counts)}")
    
    # 4. 测试 ES 检索
    print("\n[ES] 测试关键词检索")
    search_results = await es_repo.multi_search(
        query=["北京", "旅游"],
        user_id="default",
        size=10,
    )
    print(f"  - 查询'北京 旅游': 找到 {len(search_results)} 条")
    
    if search_results:
        print(f"  - 第一条结果:")
        first = search_results[0]['_source']
        print(f"    type: {first.get('type')}")
        print(f"    episode: {first.get('episode', '')[:100]}...")
    
    # 5. 结果汇总
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    print(f"\n✅ MongoDB MemCell: {len(memcells)} 个")
    print(f"✅ Milvus 记录: {len(milvus_results)} 条")
    print(f"✅ ES 记录: {len(es_results)} 条")
    print(f"✅ ES 检索: {len(search_results)} 条（关键词'北京 旅游'）")
    
    if len(milvus_results) > 0 and len(es_results) > 0 and len(search_results) > 0:
        print("\n🎉 测试通过！V3 API 功能正常！")
    else:
        print("\n⚠️ 部分功能异常，请检查日志")


async def main():
    """主函数"""
    print("=" * 80)
    print("V3 API 简单测试（同款 extract_memory 功能）")
    print("=" * 80)
    print(f"数据文件: {DATA_FILE}")
    print("=" * 80)
    
    # 步骤 1: 清空数据
    await clear_all_data()
    
    # 步骤 2: 提取记忆
    count = await extract_memories()
    
    # 等待数据刷新
    print("\n⏳ 等待 3 秒，确保数据写入...")
    await asyncio.sleep(3)
    
    # 步骤 3: 验证结果
    await verify_results(count)
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

