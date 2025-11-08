"""MemCell 到 Milvus 同步服务

负责将 MemCell 拆分并同步到 Milvus 向量数据库。
一个 MemCell 会被拆分成多条记录：
- episode (1条)
- semantic_memories (N条)
- event_log (0-1条)
"""

from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from agentic_layer.vectorize_service import DeepInfraVectorizeServiceInterface
from core.di import get_bean_by_type, service
from common_utils.datetime_utils import get_now_with_timezone

logger = logging.getLogger(__name__)


@service(name="memcell_milvus_sync_service", primary=True)
class MemCellMilvusSyncService:
    """MemCell 到 Milvus 同步服务
    
    将 MemCell 拆分成多条记录存储到 Milvus：
    1. episode - 情景记忆主体
    2. semantic_memories - 每个语义记忆单独一条
    3. event_log - 事件日志（如果有）
    """

    def __init__(
        self,
        milvus_repo: Optional[EpisodicMemoryMilvusRepository] = None,
        es_repo: Optional[EpisodicMemoryEsRepository] = None,
        vectorize_service: Optional[DeepInfraVectorizeServiceInterface] = None,
    ):
        """初始化同步服务
        
        Args:
            milvus_repo: Milvus 仓库实例（可选，不提供则从 DI 获取）
            es_repo: ES 仓库实例（可选，不提供则从 DI 获取）
            vectorize_service: 向量化服务实例（可选，不提供则从 DI 获取）
        """
        self.milvus_repo = milvus_repo or get_bean_by_type(
            EpisodicMemoryMilvusRepository
        )
        self.es_repo = es_repo or get_bean_by_type(EpisodicMemoryEsRepository)
        
        if vectorize_service is None:
            from agentic_layer.vectorize_service import get_vectorize_service
            self.vectorize_service = get_vectorize_service()
        else:
            self.vectorize_service = vectorize_service
        
        logger.info("MemCellMilvusSyncService 初始化完成（支持 Milvus + ES）")

    async def sync_memcell(
        self, memcell: MemCell, sync_to_es: bool = True, sync_to_milvus: bool = True
    ) -> Dict[str, int]:
        """同步单个 MemCell 到 Milvus 和 ES
        
        Args:
            memcell: MemCell 文档对象
            sync_to_es: 是否同步到 ES（默认 True）
            sync_to_milvus: 是否同步到 Milvus（默认 True）
            
        Returns:
            同步统计信息 {"episode": 1, "semantic_memory": N, "event_log": 0/1}
        """
        stats = {"episode": 0, "semantic_memory": 0, "event_log": 0, "es_records": 0}
        
        try:
            # 同步到 Milvus
            if sync_to_milvus:
                # 1. 同步 episode
                if hasattr(memcell, 'episode') and memcell.episode:
                    await self._sync_episode(memcell)
                    stats["episode"] = 1
                    logger.debug(f"✅ 已同步 episode 到 Milvus: {memcell.event_id}")
                
                # 2. 同步 semantic_memories
                if hasattr(memcell, 'semantic_memories') and memcell.semantic_memories:
                    count = await self._sync_semantic_memories(memcell)
                    stats["semantic_memory"] = count
                    logger.debug(f"✅ 已同步 {count} 个 semantic_memories 到 Milvus: {memcell.event_id}")
                
                # 3. 同步 event_log（多个 atomic_fact）
                if hasattr(memcell, 'event_log') and memcell.event_log:
                    count = await self._sync_event_log(memcell)
                    stats["event_log"] = count
                    logger.debug(f"✅ 已同步 {count} 个 event_log atomic_facts 到 Milvus: {memcell.event_id}")
                
                # 强制刷新 Milvus，确保数据写入
                await self.milvus_repo.flush()
                logger.debug(f"✅ Milvus 数据已刷新: {memcell.event_id}")
            
            # 同步到 ES
            if sync_to_es:
                es_count = await self._sync_to_es(memcell)
                stats["es_records"] = es_count
                logger.debug(f"✅ 已同步 {es_count} 条记录到 ES: {memcell.event_id}")
                
                # 刷新 ES 索引，确保数据可搜索
                # 注意：ES 的 refresh 是索引级别的，不是仓库方法
                # 这里手动刷新（生产环境可能不需要，ES 默认 1 秒自动刷新）
                try:
                    client = await self.es_repo.get_client()
                    index_name = self.es_repo.get_index_name()
                    await client.indices.refresh(index=index_name)
                    logger.debug(f"✅ ES 索引已刷新: {index_name}")
                except Exception as e:
                    logger.warning(f"ES 索引刷新失败（可能不影响使用）: {e}")
            
            logger.info(
                f"MemCell 同步完成: {memcell.event_id}, 统计: {stats}"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"MemCell 同步失败: {memcell.event_id}, error={e}")
            raise

    async def _sync_episode(self, memcell: MemCell) -> None:
        """同步 episode 到 Milvus
        
        Args:
            memcell: MemCell 文档对象
        """
        # 从 MongoDB 读取 embedding（必须存在）
        vector = None
        if hasattr(memcell, 'extend') and memcell.extend and 'embedding' in memcell.extend:
            vector = memcell.extend['embedding']
            # 确保是 list 格式（可能是 numpy array）
            if hasattr(vector, 'tolist'):
                vector = vector.tolist()
            logger.debug(f"从 MongoDB 读取 episode embedding: {memcell.event_id}")
        
        if not vector:
            logger.warning(
                f"episode 缺少 embedding，跳过同步到 Milvus: {memcell.event_id}"
            )
            return
        
        # 准备搜索内容
        search_content = []
        if hasattr(memcell, 'subject') and memcell.subject:
            search_content.append(memcell.subject)
        if hasattr(memcell, 'summary') and memcell.summary:
            search_content.append(memcell.summary)
        if not search_content:
            search_content.append(memcell.episode[:100])  # 使用 episode 前 100 字符
        
        # 确保 vector 是 list 格式
        if hasattr(vector, 'tolist'):
            vector = vector.tolist()
        
        # 存储
        await self.milvus_repo.create_and_save_episodic_memory(
            event_id=f"{str(memcell.event_id)}_episode",
            user_id=memcell.user_id,
            timestamp=memcell.timestamp or get_now_with_timezone(),
            episode=memcell.episode,
            search_content=search_content,
            vector=vector,
            memory_sub_type="episode",
            event_type=str(memcell.type) if hasattr(memcell, 'type') else "conversation",
            user_name=memcell.user_id,
            title=getattr(memcell, 'subject', None),
            summary=getattr(memcell, 'summary', None),
            group_id=getattr(memcell, 'group_id', None),
            participants=getattr(memcell, 'participants', None),
            subject=getattr(memcell, 'subject', None),
            keywords=getattr(memcell, 'keywords', None),
            linked_entities=getattr(memcell, 'linked_entities', None),
            memcell_event_id_list=[str(memcell.event_id)],
            extend=getattr(memcell, 'extend', None),
            parent_event_id=str(memcell.event_id),  # 父事件ID，转成字符串
        )

    async def _sync_semantic_memories(self, memcell: MemCell) -> int:
        """同步 semantic_memories 到 Milvus
        
        Args:
            memcell: MemCell 文档对象
            
        Returns:
            同步的 semantic_memory 数量
        """
        count = 0
        
        for i, sem_mem in enumerate(memcell.semantic_memories):
            try:
                # 获取 embedding（可能是对象属性或字典键）
                if isinstance(sem_mem, dict):
                    embedding = sem_mem.get('embedding')
                    content = sem_mem.get('content', str(sem_mem))
                else:
                    embedding = getattr(sem_mem, 'embedding', None)
                    content = getattr(sem_mem, 'content', str(sem_mem))
                
                # 检查是否有向量
                if not embedding or (isinstance(embedding, list) and len(embedding) == 0):
                    logger.warning(
                        f"semantic_memory 没有向量，跳过: {memcell.event_id}_semantic_{i}"
                    )
                    continue
                
                # 提取其他字段
                if isinstance(sem_mem, dict):
                    start_time_str = sem_mem.get('start_time')
                    end_time_str = sem_mem.get('end_time')
                    duration_days = sem_mem.get('duration_days')
                    source_episode_id = sem_mem.get('source_episode_id')
                else:
                    start_time_str = getattr(sem_mem, 'start_time', None)
                    end_time_str = getattr(sem_mem, 'end_time', None)
                    duration_days = getattr(sem_mem, 'duration_days', None)
                    source_episode_id = getattr(sem_mem, 'source_episode_id', None)
                
                # 转换日期字符串为 datetime
                start_time_dt = None
                end_time_dt = None
                if start_time_str:
                    try:
                        from common_utils.datetime_utils import from_iso_format
                        if isinstance(start_time_str, str):
                            start_time_dt = from_iso_format(start_time_str + "T00:00:00+08:00")
                    except Exception:
                        pass
                if end_time_str:
                    try:
                        from common_utils.datetime_utils import from_iso_format
                        if isinstance(end_time_str, str):
                            end_time_dt = from_iso_format(end_time_str + "T23:59:59+08:00")
                    except Exception:
                        pass
                
                # 存储
                await self.milvus_repo.create_and_save_episodic_memory(
                    event_id=f"{str(memcell.event_id)}_semantic_{i}",
                    user_id=memcell.user_id,
                    timestamp=memcell.timestamp or get_now_with_timezone(),
                    episode=content,
                    search_content=[content],
                    vector=embedding,
                    memory_sub_type="semantic_memory",
                    event_type=str(memcell.type) if hasattr(memcell, 'type') else "conversation",
                    group_id=getattr(memcell, 'group_id', None),
                    participants=getattr(memcell, 'participants', None),
                    start_time=start_time_dt,  # 语义记忆开始时间
                    end_time=end_time_dt,      # 语义记忆结束时间
                    extend={
                        "duration_days": duration_days,
                        "source_episode_id": source_episode_id,
                    },
                    parent_event_id=str(memcell.event_id),
                )
                count += 1
                
            except Exception as e:
                logger.error(
                    f"同步 semantic_memory 失败: {memcell.event_id}_semantic_{i}, error={e}"
                )
                continue
        
        return count

    async def _sync_event_log(self, memcell: MemCell) -> int:
        """同步 event_log 到 Milvus
        
        event_log 包含多个 atomic_fact，每个都单独存储一条记录
        
        Args:
            memcell: MemCell 文档对象
            
        Returns:
            同步的 atomic_fact 数量
        """
        count = 0
        
        try:
            event_log = memcell.event_log
            
            # 提取 atomic_fact 列表和 embeddings
            atomic_facts = []
            fact_embeddings = []
            event_log_time = None
            
            if hasattr(event_log, 'to_dict'):
                event_log_dict = event_log.to_dict()
                atomic_facts = event_log_dict.get('atomic_fact', [])
                fact_embeddings = event_log_dict.get('fact_embeddings', [])
                event_log_time = event_log_dict.get('time', None)
            elif isinstance(event_log, dict):
                atomic_facts = event_log.get('atomic_fact', [])
                fact_embeddings = event_log.get('fact_embeddings', [])
                event_log_time = event_log.get('time', None)
            else:
                logger.warning(f"event_log 格式不支持，跳过: {memcell.event_id}")
                return 0
            
            # 检查是否有 atomic_fact
            if not atomic_facts:
                logger.warning(f"event_log 没有 atomic_fact，跳过: {memcell.event_id}")
                return 0
            
            # 检查是否有 embeddings
            if not fact_embeddings or len(fact_embeddings) != len(atomic_facts):
                logger.warning(
                    f"event_log 缺少 fact_embeddings 或数量不匹配，跳过: {memcell.event_id} "
                    f"(facts={len(atomic_facts)}, embeddings={len(fact_embeddings) if fact_embeddings else 0})"
                )
                return 0
            
            # 为每个 atomic_fact 单独存储
            for i, fact in enumerate(atomic_facts):
                try:
                    # 从 MongoDB 读取 embedding
                    vector = fact_embeddings[i]
                    
                    # 确保是 list 格式
                    if hasattr(vector, 'tolist'):
                        vector = vector.tolist()
                    
                    # 存储
                    await self.milvus_repo.create_and_save_episodic_memory(
                        event_id=f"{str(memcell.event_id)}_eventlog_{i}",
                        user_id=memcell.user_id,
                        timestamp=memcell.timestamp or get_now_with_timezone(),
                        episode=fact,  # 使用 atomic_fact 作为 episode
                        search_content=[fact],
                        vector=vector,
                        memory_sub_type="event_log",
                        event_type=str(memcell.type) if hasattr(memcell, 'type') else "conversation",
                        group_id=getattr(memcell, 'group_id', None),
                        participants=getattr(memcell, 'participants', None),
                        extend={
                            "event_log_time": event_log_time,
                            "atomic_fact_index": i,
                            "total_atomic_facts": len(atomic_facts),
                        },
                        parent_event_id=str(memcell.event_id),
                    )
                    count += 1
                    
                except Exception as e:
                    logger.error(
                        f"同步 atomic_fact 失败: {memcell.event_id}_eventlog_{i}, error={e}"
                    )
                    continue
            
            return count
            
        except Exception as e:
            logger.error(f"同步 event_log 失败: {memcell.event_id}, error={e}")
            raise

    async def _sync_to_es(self, memcell: MemCell) -> int:
        """同步 MemCell 到 ES（拆分存储）
        
        和 Milvus 一样，拆分成多条记录：
        - 1 条 episode
        - N 条 semantic_memory
        - M 条 event_log (atomic_fact)
        
        Args:
            memcell: MemCell 文档对象
            
        Returns:
            同步的记录数量
        """
        count = 0
        
        try:
            # 1. 同步 episode
            if hasattr(memcell, 'episode') and memcell.episode:
                search_content = []
                if hasattr(memcell, 'subject') and memcell.subject:
                    search_content.append(memcell.subject)
                if hasattr(memcell, 'summary') and memcell.summary:
                    search_content.append(memcell.summary)
                search_content.append(memcell.episode[:500])
                
                await self.es_repo.create_and_save_episodic_memory(
                    event_id=f"{str(memcell.event_id)}_episode",
                    user_id=memcell.user_id,
                    timestamp=memcell.timestamp or get_now_with_timezone(),
                    episode=memcell.episode,
                    search_content=search_content,
                    user_name=memcell.user_id,
                    title=getattr(memcell, 'subject', None),
                    summary=getattr(memcell, 'summary', None),
                    group_id=getattr(memcell, 'group_id', None),
                    participants=getattr(memcell, 'participants', None),
                    event_type="episode",  # 标记类型
                    subject=getattr(memcell, 'subject', None),
                    memcell_event_id_list=[str(memcell.event_id)],
                )
                count += 1
            
            # 2. 同步 semantic_memories
            if hasattr(memcell, 'semantic_memories') and memcell.semantic_memories:
                for i, sem_mem in enumerate(memcell.semantic_memories):
                    try:
                        if isinstance(sem_mem, dict):
                            content = sem_mem.get('content', '')
                        else:
                            content = getattr(sem_mem, 'content', '')
                        
                        if not content:
                            continue
                        
                        await self.es_repo.create_and_save_episodic_memory(
                            event_id=f"{str(memcell.event_id)}_semantic_{i}",
                            user_id=memcell.user_id,
                            timestamp=memcell.timestamp or get_now_with_timezone(),
                            episode=content,
                            search_content=[content],
                            group_id=getattr(memcell, 'group_id', None),
                            participants=getattr(memcell, 'participants', None),
                            event_type="semantic_memory",  # 标记类型
                            memcell_event_id_list=[str(memcell.event_id)],
                        )
                        count += 1
                    except Exception as e:
                        logger.error(f"同步 semantic_memory 到 ES 失败: {i}, error={e}")
                        continue
            
            # 3. 同步 event_log (atomic_facts)
            if hasattr(memcell, 'event_log') and memcell.event_log:
                event_log = memcell.event_log
                
                if isinstance(event_log, dict):
                    atomic_facts = event_log.get('atomic_fact', [])
                elif hasattr(event_log, 'atomic_fact'):
                    atomic_facts = event_log.atomic_fact
                else:
                    atomic_facts = []
                
                for i, fact in enumerate(atomic_facts):
                    try:
                        await self.es_repo.create_and_save_episodic_memory(
                            event_id=f"{str(memcell.event_id)}_eventlog_{i}",
                            user_id=memcell.user_id,
                            timestamp=memcell.timestamp or get_now_with_timezone(),
                            episode=fact,
                            search_content=[fact],
                            group_id=getattr(memcell, 'group_id', None),
                            participants=getattr(memcell, 'participants', None),
                            event_type="event_log",  # 标记类型
                            memcell_event_id_list=[str(memcell.event_id)],
                        )
                        count += 1
                    except Exception as e:
                        logger.error(f"同步 atomic_fact 到 ES 失败: {i}, error={e}")
                        continue
            
            return count
            
        except Exception as e:
            logger.error(f"同步到 ES 失败: {memcell.event_id}, error={e}")
            return 0

    async def sync_memcells_batch(self, memcells: List[MemCell]) -> Dict[str, Any]:
        """批量同步 MemCells 到 Milvus
        
        Args:
            memcells: MemCell 文档对象列表
            
        Returns:
            批量同步统计信息
        """
        total_stats = {
            "total_memcells": len(memcells),
            "success_memcells": 0,
            "failed_memcells": 0,
            "total_episode": 0,
            "total_semantic_memory": 0,
            "total_event_log": 0,
        }
        
        for memcell in memcells:
            try:
                stats = await self.sync_memcell(memcell)
                total_stats["success_memcells"] += 1
                total_stats["total_episode"] += stats["episode"]
                total_stats["total_semantic_memory"] += stats["semantic_memory"]
                total_stats["total_event_log"] += stats["event_log"]
            except Exception as e:
                logger.error(f"批量同步失败: {memcell.event_id}, error={e}")
                total_stats["failed_memcells"] += 1
                continue
        
        logger.info(f"批量同步完成: {total_stats}")
        return total_stats


def get_memcell_milvus_sync_service() -> MemCellMilvusSyncService:
    """获取 MemCell Milvus 同步服务实例
    
    通过依赖注入框架获取服务实例，支持单例模式。
    """
    from core.di import get_bean
    return get_bean("memcell_milvus_sync_service")

