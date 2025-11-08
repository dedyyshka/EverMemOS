import random
import time
from memory_layer.memory_manager import MemorizeRequest, MemorizeOfflineRequest
from memory_layer.memory_manager import MemoryManager
from memory_layer.types import MemoryType, MemCell, Memory, RawDataType
from infra_layer.adapters.out.persistence.document.memory.memcell import DataTypeEnum
from memory_layer.memory_extractor.profile_memory_extractor import (
    ProfileMemory,
    ProfileMemoryExtractor,
    ProfileMemoryExtractRequest,
    ProfileMemoryMerger,
    ProjectInfo,
)
from memory_layer.memory_extractor.group_profile_memory_extractor import (
    GroupProfileMemoryExtractor,
    GroupProfileMemoryExtractRequest,
    GroupProfileMemory,
)
from core.di import get_bean_by_type, enable_mock_mode, scan_packages
from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
    EpisodicMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
    ConversationStatusRawRepository,
)
from infra_layer.adapters.out.persistence.repository.core_memory_raw_repository import (
    CoreMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.repository.group_user_profile_memory_raw_repository import (
    GroupUserProfileMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.group_profile_raw_repository import (
    GroupProfileRawRepository,
)
from biz_layer.conversation_data_repo import ConversationDataRepository
from memory_layer.types import RawDataType
from typing import List, Dict, Optional
import uuid
from datetime import datetime, timedelta
import os
import asyncio
from collections import defaultdict
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from memory_layer.memcell_extractor.base_memcell_extractor import StatusResult
import traceback

from core.lock.redis_distributed_lock import distributed_lock
from core.observation.logger import get_logger
from infra_layer.adapters.out.search.elasticsearch.converter.episodic_memory_converter import (
    EpisodicMemoryConverter,
)
from infra_layer.adapters.out.search.milvus.converter.episodic_memory_milvus_converter import (
    EpisodicMemoryMilvusConverter,
)
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from biz_layer.memcell_milvus_sync import MemCellMilvusSyncService

logger = get_logger(__name__)


def _convert_data_type_to_raw_data_type(data_type) -> RawDataType:
    """
    将不同的数据类型枚举转换为统一的RawDataType

    Args:
        data_type: 可能是DataTypeEnum、RawDataType或字符串

    Returns:
        RawDataType: 转换后的统一数据类型
    """
    if isinstance(data_type, RawDataType):
        return data_type

    # 获取字符串值
    if hasattr(data_type, 'value'):
        type_str = data_type.value
    else:
        type_str = str(data_type)

    # 映射转换
    type_mapping = {
        "Conversation": RawDataType.CONVERSATION,
        "CONVERSATION": RawDataType.CONVERSATION,
        # 其他类型映射到CONVERSATION作为默认值
    }

    return type_mapping.get(type_str, RawDataType.CONVERSATION)


from biz_layer.mem_db_operations import (
    _convert_timestamp_to_time,
    _convert_episode_memory_to_doc,
    _save_memcell_to_database,
    _save_profile_memory_to_core,
    ConversationStatus,
    _update_status_for_new_conversation,
    _update_status_for_continuing_conversation,
    _update_status_after_memcell_extraction,
    _convert_original_data_for_profile_extractor,
    _save_group_profile_memory,
    _get_user_organization,
    _save_profile_memory_to_group_user_profile_memory,
    _convert_document_to_group_importance_evidence,
    _get_raw_data_by_time_range,
    _normalize_datetime_for_storage,
    _convert_projects_participated_list,
    _convert_group_profile_raw_to_memory_format,
)


def if_memorize(memcells: List[MemCell]) -> bool:
    return True


def extract_message_time(raw_data):
    """
    从RawData对象中提取消息时间

    Args:
        raw_data: RawData对象

    Returns:
        datetime: 消息时间，如果无法提取则返回None
    """
    # 优先从timestamp字段获取
    if hasattr(raw_data, 'timestamp') and raw_data.timestamp:
        try:
            return _normalize_datetime_for_storage(raw_data.timestamp)
        except Exception as e:
            logger.debug(f"Failed to parse timestamp from raw_data.timestamp: {e}")
            pass

    # 从extend字段获取
    if (
        hasattr(raw_data, 'extend')
        and raw_data.extend
        and isinstance(raw_data.extend, dict)
    ):
        timestamp_val = raw_data.extend.get('timestamp')
        if timestamp_val:
            try:
                return _normalize_datetime_for_storage(timestamp_val)
            except Exception as e:
                logger.debug(f"Failed to parse timestamp from extend field: {e}")
                pass

    return None


from core.observation.tracing.decorators import trace_logger


@trace_logger(operation_name="mem_memorize preprocess_conv_request", log_level="info")
async def preprocess_conv_request(
    request: MemorizeRequest, current_time: datetime
) -> MemorizeRequest:

    # load status table， 重新读取部分历史数据，覆盖history_raw_data_list和new_raw_data_list
    logger.info("开始处理状态表逻辑...")

    # 获取Repository实例
    try:

        status_repo = get_bean_by_type(ConversationStatusRawRepository)

        logger.info("成功获取状态表和数据Repository")
    except Exception as e:
        logger.error(f"获取Repository失败，使用原逻辑: {e}")
        traceback.print_exc()
        # 如果无法获取Repository，继续使用原有逻辑
    if not request.new_raw_data_list:
        logger.info("[mem_memorize] 没有新数据需要处理")
        return None
    else:
        # 1. 获取当前对话状态
        # 查询对话状态，真实repository返回DocConversationStatus
        doc_status = await status_repo.get_by_group_id(request.group_id)
        logger.debug(f"[mem_memorize] doc_status: {doc_status}")

        # 转换为业务层模型
        conversation_status = None
        if doc_status:
            conversation_status = ConversationStatus(
                group_id=doc_status.group_id,
                old_msg_start_time=_convert_timestamp_to_time(
                    doc_status.old_msg_start_time
                ),
                new_msg_start_time=_convert_timestamp_to_time(
                    doc_status.new_msg_start_time
                ),
                last_memcell_time=_convert_timestamp_to_time(
                    doc_status.last_memcell_time
                ),
                created_at=(
                    doc_status.created_at.isoformat()
                    if hasattr(doc_status, 'created_at') and doc_status.created_at
                    else to_iso_format(current_time)
                ),
                updated_at=(
                    doc_status.updated_at.isoformat()
                    if hasattr(doc_status, 'updated_at') and doc_status.updated_at
                    else to_iso_format(current_time)
                ),
            )


        # 3. 根据状态表决定如何读取历史数据
        history_raw_data_list = request.history_raw_data_list
        new_raw_data_list = request.new_raw_data_list

        if conversation_status:
            # 存在状态记录，根据状态决定数据范围
            logger.info(f"[mem_memorize] 找到对话状态，重新构建数据范围")

            # 获取old_msg_start_time和new_msg_start_time作为分界点
            old_msg_start_time = _normalize_datetime_for_storage(
                conversation_status.old_msg_start_time
            )
            new_msg_start_time = _normalize_datetime_for_storage(
                conversation_status.new_msg_start_time
            )

            # 检查new_raw_data_list中最早的消息时间，如果比current new_msg_start_time更早，则调整时间边界
            # 这是为了解决kafka输入不完全按顺序的问题
            if request.new_raw_data_list and new_msg_start_time:
                # 找到最早的消息时间
                earliest_new_message_time = None
                for raw_data in request.new_raw_data_list:
                    message_time = extract_message_time(raw_data)
                    if message_time and (
                        earliest_new_message_time is None
                        or message_time < earliest_new_message_time
                    ):
                        earliest_new_message_time = message_time

                # 转换new_msg_start_time为datetime对象并比较
                if earliest_new_message_time:

                    # 如果最早消息时间比当前new_msg_start_time更早，则调整时间边界
                    if (
                        new_msg_start_time
                        and earliest_new_message_time < new_msg_start_time
                    ):
                        logger.debug(
                            f"[mem_memorize] 检测到更早的消息: {earliest_new_message_time} < {new_msg_start_time}"
                        )

                        # 调整new_msg_start_time为最早消息时间
                        new_msg_start_time = earliest_new_message_time

                        # 调整old_msg_start_time为min(原old_msg_start_time, new_msg_start_time - 1ms)
                        new_boundary = earliest_new_message_time - timedelta(
                            milliseconds=1
                        )
                        old_msg_start_time = min(old_msg_start_time, new_boundary)

                        logger.debug(
                            f"[mem_memorize] 时间边界已调整: old_msg_start_time={old_msg_start_time}, new_msg_start_time={new_msg_start_time}"
                        )

                        # 时间边界调整后，更新conversation status表
                        try:
                            update_data = {
                                "old_msg_start_time": _normalize_datetime_for_storage(
                                    old_msg_start_time
                                ),
                                "new_msg_start_time": _normalize_datetime_for_storage(
                                    new_msg_start_time
                                ),
                                "updated_at": current_time,
                            }

                            result = await status_repo.upsert_by_group_id(
                                request.group_id, update_data
                            )
                            if result:
                                logger.debug(
                                    f"[mem_memorize] 时间边界调整后，conversation status表更新成功"
                                )
                            else:
                                logger.debug(
                                    f"[mem_memorize] 时间边界调整后，conversation status表更新失败"
                                )
                        except Exception as e:
                            logger.debug(
                                f"时间边界调整后，更新conversation status表异常: {e}"
                            )

            # 读取历史数据：从old_msg_start_time到new_msg_start_time 前闭后开
            now = time.time()
            history_data = []
            if new_msg_start_time:
                history_data = await _get_raw_data_by_time_range(
                    request.group_id,
                    start_time=_normalize_datetime_for_storage(old_msg_start_time),
                    end_time=_normalize_datetime_for_storage(new_msg_start_time),
                    limit=500,  # 限制历史消息数量
                )
            # 移除高频调试日志
            # 读取新数据：从new_msg_start_time到当前时间 +1ms是为了调整为前闭后闭
            new_data = []
            if new_msg_start_time:
                new_data = await _get_raw_data_by_time_range(
                    request.group_id,
                    start_time=_normalize_datetime_for_storage(new_msg_start_time),
                    end_time=_normalize_datetime_for_storage(current_time)
                    + timedelta(milliseconds=1),  # 添加结束时间为当前时间
                    limit=500,  # 限制新消息数量
                )
            # 移除高频调试日志
            logger.info(
                f"[mem_memorize] 从状态表重新读取: 历史数据 {len(history_data)} 条, 新数据 {len(new_data)} 条"
            )

            # 重新分配数据（如果 Redis 返回空，保留原始数据）
            if history_data or new_data:
                history_raw_data_list = history_data
                new_raw_data_list = new_data
                logger.info(
                    f"[mem_memorize] 使用 Redis 数据: 历史 {len(history_raw_data_list)} 条, 新数据 {len(new_raw_data_list)} 条"
                )
            else:
                history_raw_data_list = request.history_raw_data_list
                new_raw_data_list = request.new_raw_data_list
                logger.info(
                    f"[mem_memorize] Redis 无数据，保留原始请求: 历史 {len(history_raw_data_list)} 条, 新数据 {len(new_raw_data_list)} 条"
                )

        else:
            # 新对话，创建状态记录
            logger.info(f"[mem_memorize] 新对话，创建状态记录")

            # 获取最早消息时间
            earliest_new_time = _convert_timestamp_to_time(current_time, current_time)
            if request.new_raw_data_list:
                first_msg = request.new_raw_data_list[0]
                if hasattr(first_msg, 'content') and isinstance(
                    first_msg.content, dict
                ):
                    earliest_new_time = first_msg.content.get(
                        'timestamp', earliest_new_time
                    )
                elif hasattr(first_msg, 'timestamp'):
                    earliest_new_time = first_msg.timestamp

            # 使用封装函数创建新对话状态
            await _update_status_for_new_conversation(
                status_repo, request, earliest_new_time, current_time
            )
        # 4. 检查是否有数据需要处理
        if not new_raw_data_list:
            logger.info(f"[mem_memorize] 没有新数据需要处理")
            return None

        # 更新request的数据
        request.history_raw_data_list = history_raw_data_list
        request.new_raw_data_list = new_raw_data_list
    return request


async def update_status_when_no_memcell(
    request: MemorizeRequest,
    status_result: StatusResult,
    current_time: datetime,
    data_type: RawDataType,
):
    if data_type == RawDataType.CONVERSATION:
        # 尝试更新状态表
        try:
            status_repo = get_bean_by_type(ConversationStatusRawRepository)

            if status_result.should_wait:
                logger.info(f"[mem_memorize] 判断为无法判断边界继续等待，不更新状态表")
                return
            else:
                logger.info(f"[mem_memorize] 判断为非边界，继续累积msg，更新状态表")
                # 获取最新消息时间戳
                latest_time = _convert_timestamp_to_time(current_time, current_time)
                if request.new_raw_data_list:
                    last_msg = request.new_raw_data_list[-1]
                    if hasattr(last_msg, 'content') and isinstance(
                        last_msg.content, dict
                    ):
                        latest_time = last_msg.content.get('timestamp', latest_time)
                    elif hasattr(last_msg, 'timestamp'):
                        latest_time = last_msg.timestamp

                if not latest_time:
                    latest_time = min(latest_time, current_time)

                # 使用封装函数更新对话延续状态
                await _update_status_for_continuing_conversation(
                    status_repo, request, latest_time, current_time
                )

        except Exception as e:
            logger.error(f"更新状态表失败: {e}")
    else:
        pass


async def update_status_after_memcell(
    request: MemorizeRequest,
    memcells: List[MemCell],
    current_time: datetime,
    data_type: RawDataType,
):
    if data_type == RawDataType.CONVERSATION:
        # 更新状态表中的last_memcell_time至memcells最后一个时间戳
        try:
            status_repo = get_bean_by_type(ConversationStatusRawRepository)

            # 获取MemCell的时间戳
            memcell_time = None
            if memcells and hasattr(memcells[-1], 'timestamp'):
                memcell_time = memcells[-1].timestamp
            else:
                memcell_time = current_time

            # 使用封装函数更新MemCell提取后的状态
            await _update_status_after_memcell_extraction(
                status_repo, request, memcell_time, current_time
            )

            logger.info(f"[mem_memorize] 记忆提取完成，状态表已更新")

        except Exception as e:
            logger.error(f"最终状态表更新失败: {e}")
    else:
        pass


async def save_personal_profile_memory(
    profile_memories: List[ProfileMemory], version: Optional[str] = None
):
    logger.info(f"[mem_memorize] 保存 {len(profile_memories)} 个个人档案记忆到数据库")
    # 初始化Repository实例
    core_memory_repo = get_bean_by_type(CoreMemoryRawRepository)

    # 保存个人档案记忆到GroupUserProfileMemoryRawRepository
    for profile_mem in profile_memories:
        await _save_profile_memory_to_core(profile_mem, core_memory_repo, version)
        # 移除单个操作成功日志


async def save_memories(
    memory_list: List[Memory], current_time: datetime, version: Optional[str] = None
):
    logger.info(f"[mem_memorize] 保存 {len(memory_list)} 个记忆到数据库")
    # 初始化Repository实例
    episodic_memory_repo = get_bean_by_type(EpisodicMemoryRawRepository)
    group_user_profile_memory_repo = get_bean_by_type(
        GroupUserProfileMemoryRawRepository
    )
    group_profile_raw_repo = get_bean_by_type(GroupProfileRawRepository)
    episodic_memory_milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)

    # 按memory_type分类保存
    episode_memories = [
        m for m in memory_list if m.memory_type == MemoryType.EPISODE_SUMMARY
    ]
    profile_memories = [m for m in memory_list if m.memory_type == MemoryType.PROFILE]
    group_profile_memories = [
        m for m in memory_list if m.memory_type == MemoryType.GROUP_PROFILE
    ]

    # 保存情景记忆到 EpisodicMemoryRawRepository（包括 ES/Milvus）
    for episode_mem in episode_memories:
        # 转换为EpisodicMemory文档格式
        doc = _convert_episode_memory_to_doc(episode_mem, current_time)
        doc = await episodic_memory_repo.append_episodic_memory(doc)
        episode_mem.event_id = str(doc.event_id)
        
        # 保存到 ES
        es_doc = EpisodicMemoryConverter.from_mongo(doc)
        await es_doc.save()
        
        # 保存到 Milvus（添加缺失的字段）
        milvus_entity = EpisodicMemoryMilvusConverter.from_mongo(doc)
        vector = milvus_entity.get("vector") if isinstance(milvus_entity, dict) else None
        
        if not vector or (isinstance(vector, list) and len(vector) == 0):
            logger.warning(
                "[mem_memorize] 跳过写入Milvus：向量为空或缺失，event_id=%s",
                getattr(doc, 'event_id', None),
            )
        else:
            # ⚠️ 旧 converter 缺少字段，手动补全
            milvus_entity["memory_sub_type"] = "episode"  # 标记为 episode 类型
            milvus_entity["start_time"] = 0
            milvus_entity["end_time"] = 0
            # 字段名修正：旧 schema 是 detail → 新 schema 是 metadata
            if "detail" in milvus_entity:
                milvus_entity["metadata"] = milvus_entity.pop("detail")
            else:
                milvus_entity["metadata"] = "{}"
            # 确保 search_content 字段存在
            if "search_content" not in milvus_entity:
                milvus_entity["search_content"] = milvus_entity.get("episode", "")[:500]
            await episodic_memory_milvus_repo.insert(milvus_entity)
        
        logger.debug(f"✅ 保存 episode_memory: {episode_mem.event_id}")

    # 保存Profile记忆到CoreMemoryRawRepository
    for profile_mem in profile_memories:
        try:
            await _save_profile_memory_to_group_user_profile_memory(
                profile_mem, group_user_profile_memory_repo, version
            )
        except Exception as e:
            logger.error(f"保存Profile记忆失败: {e}")

    for group_profile_mem in group_profile_memories:
        try:
            await _save_group_profile_memory(
                group_profile_mem, group_profile_raw_repo, version
            )
        except Exception as e:
            logger.error(f"保存Group Profile记忆失败: {e}")

    logger.info(f"[mem_memorize] 保存完成:")
    logger.info(f"  - EPISODE_SUMMARY: {len(episode_memories)} 个")
    logger.info(f"  - PROFILE: {len(profile_memories)} 个")
    logger.info(f"  - GROUP_PROFILE: {len(group_profile_memories)} 个")


async def load_core_memories(
    request: MemorizeRequest, participants: List[str], current_time: datetime
):
    logger.info(f"[mem_memorize] 读取用户数据: {participants}")
    # 初始化Repository实例
    core_memory_repo = get_bean_by_type(CoreMemoryRawRepository)

    # 读取用户CoreMemory数据
    user_core_memories = {}
    for user_id in participants:
        try:
            core_memory = await core_memory_repo.get_by_user_id(user_id)
            if core_memory:
                user_core_memories[user_id] = core_memory
            # 移除单个用户的成功/失败日志
        except Exception as e:
            logger.error(f"获取用户 {user_id} CoreMemory失败: {e}")

    logger.info(f"[mem_memorize] 获取到 {len(user_core_memories)} 个用户CoreMemory")

    # 直接从CoreMemory转换为ProfileMemory对象列表
    old_memory_list = []
    if user_core_memories:
        for user_id, core_memory in user_core_memories.items():
            if core_memory:
                # 直接创建ProfileMemory对象
                profile_memory = ProfileMemory(
                    # Memory 基类必需字段
                    memory_type=MemoryType.CORE,
                    user_id=user_id,
                    timestamp=to_iso_format(current_time),
                    ori_event_id_list=[],
                    # Memory 基类可选字段
                    subject=f"{getattr(core_memory, 'user_name', user_id)}的个人档案",
                    summary=f"用户{user_id}的基本信息：{getattr(core_memory, 'position', '未知角色')}",
                    group_id=request.group_id,
                    participants=[user_id],
                    type=RawDataType.CONVERSATION,
                    # ProfileMemory 特有字段 - 直接使用原始字典格式
                    hard_skills=getattr(core_memory, 'hard_skills', None),
                    soft_skills=getattr(core_memory, 'soft_skills', None),
                    output_reasoning=getattr(core_memory, 'output_reasoning', None),
                    motivation_system=getattr(core_memory, 'motivation_system', None),
                    fear_system=getattr(core_memory, 'fear_system', None),
                    value_system=getattr(core_memory, 'value_system', None),
                    humor_use=getattr(core_memory, 'humor_use', None),
                    colloquialism=getattr(core_memory, 'colloquialism', None),
                    projects_participated=_convert_projects_participated_list(
                        getattr(core_memory, 'projects_participated', None)
                    ),
                )
                old_memory_list.append(profile_memory)

        logger.info(
            f"[mem_memorize] 直接转换了 {len(old_memory_list)} 个CoreMemory为ProfileMemory"
        )
    else:
        logger.info(f"[mem_memorize] 没有用户CoreMemory数据，old_memory_list为空")


async def memorize(request: MemorizeRequest) -> List[Memory]:

    # logger.info(f"[mem_memorize] request: {request}")

    # logger.info(f"[mem_memorize] memorize request: {request}")
    logger.info(f"[mem_memorize] request.current_time: {request.current_time}")
    # 获取当前时间，用于所有时间相关操作
    if request.current_time:
        current_time = request.current_time
    else:
        current_time = get_now_with_timezone() + timedelta(seconds=1)
    logger.info(f"[mem_memorize] 当前时间: {current_time}")

    memory_manager = MemoryManager()

    memory_types = [MemoryType.EPISODE_SUMMARY]
    if request.raw_data_type == RawDataType.CONVERSATION:
        request = await preprocess_conv_request(request, current_time)
        if request == None:
            return None

    if request.raw_data_type == RawDataType.CONVERSATION:
        # async with distributed_lock(f"memcell_extract_{request.group_id}") as acquired:
        #     # 120s等待，获取不到
        #     if not acquired:
        #         logger.warning(f"[mem_memorize] 获取分布式锁失败: {request.group_id}")
        now = time.time()
        logger.debug(
            f"[memorize memorize] 提取MemCell开始: group_id={request.group_id}, group_name={request.group_name}, "
            f"semantic_extraction={request.enable_semantic_extraction}"
        )
        memcell_result = await memory_manager.extract_memcell(
            request.history_raw_data_list,
            request.new_raw_data_list,
            request.raw_data_type,
            request.group_id,
            request.group_name,
            request.user_id_list,
            enable_semantic_extraction=request.enable_semantic_extraction,
            enable_event_log_extraction=request.enable_event_log_extraction,
        )
        logger.debug(f"[memorize memorize] 提取MemCell耗时: {time.time() - now}秒")
    else:
        now = time.time()
        logger.debug(
            f"[memorize memorize] 提取MemCell开始: group_id={request.group_id}, group_name={request.group_name}, "
            f"semantic_extraction={request.enable_semantic_extraction}, "
            f"event_log_extraction={request.enable_event_log_extraction}"
        )
        memcell_result = await memory_manager.extract_memcell(
            request.history_raw_data_list,
            request.new_raw_data_list,
            request.raw_data_type,
            request.group_id,
            request.group_name,
            request.user_id_list,
            enable_semantic_extraction=request.enable_semantic_extraction,
            enable_event_log_extraction=request.enable_event_log_extraction,
        )
        logger.debug(f"[memorize memorize] 提取MemCell耗时: {time.time() - now}秒")

    if memcell_result == None:
        logger.warning(f"[mem_memorize] 跳过提取MemCell")
        return None

    logger.debug(f"[mem_memorize] memcell_result: {memcell_result}")
    memcell, status_result = memcell_result

    if memcell == None:
        await update_status_when_no_memcell(
            request, status_result, current_time, request.raw_data_type
        )
        logger.warning(f"[mem_memorize] 跳过提取MemCell")
        return None
    else:
        logger.info(f"[mem_memorize] 成功提取MemCell")

    # TODO: 读状态表，读取累积的MemCell数据表，判断是否要做memorize计算

    # MemCell存表
    memcell = await _save_memcell_to_database(memcell, current_time)

    # 同步 MemCell 到 Milvus 和 ES（包括 episode/semantic_memories/event_log）
    memcell_repo = get_bean_by_type(MemCellRawRepository)
    doc_memcell = await memcell_repo.get_by_event_id(str(memcell.event_id))
    
    if doc_memcell:
        sync_service = get_bean_by_type(MemCellMilvusSyncService)
        sync_stats = await sync_service.sync_memcell(
            doc_memcell, 
            sync_to_es=True, 
            sync_to_milvus=True
        )
        logger.info(
            f"[mem_memorize] MemCell 同步到 Milvus/ES 完成: {memcell.event_id}, "
            f"stats={sync_stats}"
        )
    else:
        logger.warning(f"[mem_memorize] 无法加载 MemCell 进行同步: {memcell.event_id}")

    # print_memory = random.random() < 0.1

    logger.info(f"[mem_memorize] 成功保存MemCell: {memcell.event_id}")

    # if print_memory:
    #     logger.info(f"[mem_memorize] 打印MemCell: {memcell}")

    memcells = [memcell]

    # 读取记忆的流程
    participants = []
    for memcell in memcells:
        if memcell.participants:
            participants.extend(memcell.participants)

    if if_memorize(memcells):
        # 加锁
        # 使用真实Repository读取用户数据
        old_memory_list = await load_core_memories(request, participants, current_time)

        # 提取记忆
        memory_list = []
        for memory_type in memory_types:
            # 移除单个类型提取的日志
            extracted_memories = await memory_manager.extract_memory(
                memcell_list=memcells,
                memory_type=memory_type,
                user_ids=participants,
                group_id=request.group_id,
                group_name=request.group_name,
                old_memory_list=old_memory_list,
            )
            if extracted_memories:
                memory_list += extracted_memories
        # 移除详细的提取完成日志

        # if print_memory:
        #     logger.info(f"[mem_memorize] 打印记忆: {memory_list}")
        # 保存记忆到数据库
        if memory_list:
            await save_memories(memory_list, current_time)

        await update_status_after_memcell(
            request, memcells, current_time, request.raw_data_type
        )

        # TODO: 实际项目中应该加锁避免并发问题
        # 释放锁
        return memory_list
    else:
        return None


async def memorize_offline_by_group_id(
    group_id: str, memcells: List[MemCell], old_memory_list: List[Memory]
) -> List[Memory]:
    """
    按群组ID进行离线记忆处理

    并行调用群组档案记忆提取器和档案记忆提取器，处理指定群组的记忆数据。

    Args:
        group_id: 群组ID
        memcells: 该群组的MemCell列表
        old_memory_list: 该群组的历史记忆列表（包含episode_memories, profile_memories等）

    Returns:
        提取的记忆列表
    """
    logger.info(
        f"[memorize_offline_by_group_id] 开始处理群组: {group_id}, MemCell数量: {len(memcells)}, 历史记忆数量: {len(old_memory_list)}"
    )

    # 统计old_memory_list中的记忆类型
    memory_type_counts = {}
    for memory in old_memory_list:
        memory_type = (
            memory.memory_type.value
            if hasattr(memory.memory_type, 'value')
            else str(memory.memory_type)
        )
        memory_type_counts[memory_type] = memory_type_counts.get(memory_type, 0) + 1

    logger.info(
        f"[memorize_offline_by_group_id] 群组 {group_id} 历史记忆类型统计: {memory_type_counts}"
    )

    # 提取参与者ID
    participants = []
    for memcell in memcells:
        if memcell.participants:
            participants.extend(memcell.participants)
    participants = list(set(participants))  # 去重

    logger.info(
        f"[memorize_offline_by_group_id] 群组 {group_id} 参与者: {participants}"
    )

    user_organization = await _get_user_organization(participants)
    group_repo = get_bean_by_type(ChatGroupRawRepository)
    now = time.time()
    logger.debug(
        f"[memorize_offline_by_group_id] 获取群组名称开始: group_id={group_id}"
    )
    group_name = await group_repo.get_name_by_id(group_id)
    logger.debug(
        f"[memorize_offline_by_group_id] 获取群组名称耗时: {time.time() - now}秒"
    )
    # 初始化记忆提取器
    memory_manager = MemoryManager()

    # 准备并行任务
    tasks = []

    # 任务1: 群组档案记忆提取
    try:
        group_profile_task = asyncio.create_task(
            memory_manager.extract_memory(
                memcell_list=memcells,
                memory_type=MemoryType.GROUP_PROFILE,
                user_ids=participants,
                group_id=group_id,
                group_name=group_name,
                old_memory_list=old_memory_list,
                user_organization=user_organization,
            ),
            name=f"group_profile_{group_id}",
        )
        tasks.append(group_profile_task)
        logger.info(
            f"[memorize_offline_by_group_id] 已创建群组档案记忆提取任务: {group_id}"
        )

    except Exception as e:
        logger.error(f"创建群组档案记忆提取任务失败: {e}")

    # 任务2: 档案记忆提取
    try:
        profile_extract_task = asyncio.create_task(
            memory_manager.extract_memory(
                memcell_list=memcells,
                memory_type=MemoryType.PROFILE,
                user_ids=participants,
                group_id=group_id,
                group_name=group_name,
                old_memory_list=old_memory_list,
            ),
            name=f"profile_{group_id}",
        )
        tasks.append(profile_extract_task)
        logger.info(
            f"[memorize_offline_by_group_id] 已创建档案记忆提取任务: {group_id}"
        )

    except Exception as e:
        logger.error(f"创建档案记忆提取任务失败: {e}")

    # 并行执行所有任务
    if not tasks:
        logger.warning(
            f"[memorize_offline_by_group_id] 群组 {group_id} 没有可执行的任务"
        )
        return []

    logger.info(
        f"[memorize_offline_by_group_id] 开始并行执行 {len(tasks)} 个任务: {group_id}"
    )

    try:
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 整理结果
        extracted_memories = []
        for i, result in enumerate(results):
            task_name = tasks[i].get_name()
            if isinstance(result, Exception):
                logger.error(
                    f"[memorize_offline_by_group_id] 任务 {task_name} 执行失败: {result}"
                )
            elif result is None:
                logger.warning(
                    f"[memorize_offline_by_group_id] 任务 {task_name} 返回空结果"
                )
            elif isinstance(result, list):
                extracted_memories.extend(result)
                # 移除单个任务成功的日志
            else:
                logger.warning(
                    f"[memorize_offline_by_group_id] 任务 {task_name} 返回未知格式结果: {type(result)}"
                )

        logger.info(
            f"[memorize_offline_by_group_id] 群组 {group_id} 处理完成，共提取 {len(extracted_memories)} 个记忆"
        )
        return extracted_memories

    except Exception as e:
        logger.error(f"群组 {group_id} 并行执行失败: {e}")
        import traceback

        traceback.print_exc()
        return []


def get_version_from_request(request: MemorizeOfflineRequest) -> str:
    # 1. 获取 memorize_to 日期
    target_date = request.memorize_to

    # 2. 倒退一天
    previous_day = target_date - timedelta(days=1)

    # 3. 格式化为 "YYYY-MM" 字符串
    return previous_day.strftime("%Y-%m")


async def memorize_offline(request: MemorizeOfflineRequest) -> List[Memory]:
    """
    离线记忆处理功能

    根据MemorizeOfflineRequest参数读取指定时间范围内的MemCell数据，
    然后按照memorize函数中"# 读取记忆的流程"之后的流程进行记忆提取和保存。
    不进行对话状态更新。

    Args:
        request: MemorizeOfflineRequest对象，包含时间范围和group_id

    Returns:
        提取的记忆列表
    """
    logger.info(f"[memorize_offline] 离线记忆处理开始: {request}")

    # 获取当前时间，用于所有时间相关操作
    current_time = get_now_with_timezone()
    logger.info(f"[memorize_offline] 当前时间: {current_time}")

    # 获取MemCell Repository
    memcell_repo = get_bean_by_type(MemCellRawRepository)

    # 获取其他Repository实例
    episodic_memory_repo = get_bean_by_type(EpisodicMemoryRawRepository)
    core_memory_repo = get_bean_by_type(CoreMemoryRawRepository)
    group_user_profile_memory_repo = get_bean_by_type(
        GroupUserProfileMemoryRawRepository
    )
    group_profile_raw_repo = get_bean_by_type(GroupProfileRawRepository)

    # 初始化MemoryManager，用于后续记忆合并
    memory_manager = MemoryManager()

    logger.info(
        f"[memorize_offline] 开始读取MemCell数据: 时间范围 {request.memorize_from} ~ {request.memorize_to}"
    )

    try:
        # 步骤1: 先读取MemCell数据
        # import ipdb; ipdb.set_trace()
        doc_memcells = await memcell_repo.find_by_time_range(
            start_time=request.memorize_from,
            end_time=request.memorize_to,
            sort_desc=False,
        )

        logger.info(f"[memorize_offline] MemCell读取完成: {len(doc_memcells)} 条")

        if not doc_memcells:
            logger.info(f"[memorize_offline] 未发现MemCell数据，结束处理")
            return []

        # 步骤2: 从MemCell中提取所有用户ID
        all_user_ids = set()
        for memcell in doc_memcells:
            if memcell.user_id:
                all_user_ids.add(memcell.user_id)
            if hasattr(memcell, 'participants') and memcell.participants:
                all_user_ids.update(memcell.participants)

        user_id_list = list(all_user_ids)
        logger.info(f"[memorize_offline] 提取用户ID: {len(user_id_list)} 个用户")

        # 步骤3: 并行读取Episode和Profile数据
        logger.info(f"[memorize_offline] 开始并行读取Episode和Profile数据...")

        episode_task = asyncio.create_task(
            episodic_memory_repo.find_by_time_range(
                start_time=request.memorize_from,
                end_time=request.memorize_to,
                sort_desc=False,
            ),
            name="read_episodes",
        )

        profile_task = asyncio.create_task(
            core_memory_repo.find_by_user_ids(user_id_list), name="read_profiles"
        )

        doc_episodes, doc_profiles = await asyncio.gather(episode_task, profile_task)

        logger.info(f"[memorize_offline] 数据读取完成:")
        logger.info(f"  - MemCell: {len(doc_memcells)} 条")
        logger.info(f"  - Episode Memory: {len(doc_episodes)} 条")
        logger.info(f"  - Profile Memory: {len(doc_profiles)} 条")

    except Exception as e:
        logger.error(f"数据读取失败: {e}")
        import traceback

        traceback.print_exc()
        return []

    # 按group_id分组数据
    logger.info(f"[memorize_offline] 开始按group_id分组数据...")

    # 分组MemCell
    memcells_by_group: Dict[str, List[MemCell]] = defaultdict(list)
    for doc_memcell in doc_memcells:
        if doc_memcell.group_id:
            # 使用统一的转换函数处理original_data
            original_data_list = _convert_original_data_for_profile_extractor(
                doc_memcell
            )

            # 转换为业务层MemCell对象
            business_memcell = MemCell(
                event_id=doc_memcell.event_id,
                user_id_list=[doc_memcell.user_id] if doc_memcell.user_id else [],
                original_data=original_data_list,
                timestamp=(
                    doc_memcell.timestamp
                    if hasattr(doc_memcell.timestamp, 'isoformat')
                    else doc_memcell.timestamp
                ),
                summary=getattr(doc_memcell, 'summary', ''),
                group_id=doc_memcell.group_id,
                participants=getattr(
                    doc_memcell,
                    'participants',
                    [doc_memcell.user_id] if doc_memcell.user_id else [],
                ),
                type=_convert_data_type_to_raw_data_type(
                    getattr(doc_memcell, 'type', 'Conversation')
                ),
                subject=getattr(doc_memcell, 'subject', ''),
                keywords=getattr(doc_memcell, 'keywords', []),
                linked_entities=getattr(doc_memcell, 'linked_entities', []),
                episode=getattr(doc_memcell, 'episode', None),
            )
            memcells_by_group[doc_memcell.group_id].append(business_memcell)

    # 分组Episode Memory（转换为业务层Memory对象，并进行时间范围过滤）
    episodes_by_group: Dict[str, List[Memory]] = defaultdict(list)
    for doc_episode in doc_episodes:
        # 时间范围过滤
        episode_time = doc_episode.timestamp
        if (
            doc_episode.group_id
            and episode_time >= request.memorize_from
            and episode_time <= request.memorize_to
        ):
            # 转换为Memory对象
            episode_memory = Memory(
                memory_type=MemoryType.EPISODE_SUMMARY,
                user_id=doc_episode.user_id,
                timestamp=(
                    doc_episode.timestamp
                    if hasattr(doc_episode.timestamp, 'isoformat')
                    else doc_episode.timestamp
                ),
                ori_event_id_list=[],
                group_id=doc_episode.group_id,
                participants=doc_episode.participants,
                subject=doc_episode.subject,
                summary=doc_episode.summary,
                episode=doc_episode.episode,
                type=_convert_data_type_to_raw_data_type(
                    getattr(doc_episode, 'type', 'Conversation')
                ),
                keywords=doc_episode.keywords,
                linked_entities=doc_episode.linked_entities,
                extend=doc_episode.extend,
            )
            episodes_by_group[doc_episode.group_id].append(episode_memory)

    # 分组Profile Memory（根据MemCell和Episode中涉及的用户ID来分配）
    profiles_by_group: Dict[str, List[Memory]] = defaultdict(list)

    # 先收集每个群组涉及的用户ID
    group_user_mapping: Dict[str, set] = defaultdict(set)

    # 从MemCell中收集用户ID
    for group_id, memcells in memcells_by_group.items():
        for memcell in memcells:
            if memcell.user_id_list:
                group_user_mapping[group_id].update(memcell.user_id_list)
            if memcell.participants:
                group_user_mapping[group_id].update(memcell.participants)

    # 从Episode中收集用户ID
    for group_id, episodes in episodes_by_group.items():
        for episode in episodes:
            if episode.user_id:
                group_user_mapping[group_id].add(episode.user_id)
            if episode.participants:
                group_user_mapping[group_id].update(episode.participants)

    logger.info(f"[memorize_offline] 群组用户映射:")
    for group_id, user_ids in group_user_mapping.items():
        logger.info(
            f"  - {group_id}: {len(user_ids)} 个用户 {list(user_ids)[:3]}{'...' if len(user_ids) > 3 else ''}"
        )

    # 根据用户ID将Profile Memory分配到对应群组
    for doc_profile in doc_profiles:
        profile_user_id = doc_profile.user_id
        # 查找该用户属于哪些群组
        for group_id, user_ids in group_user_mapping.items():
            if profile_user_id in user_ids:
                # 将该用户的Profile Memory添加到对应群组
                # 注意：一个用户可能属于多个群组，所以可能重复添加
                profile_memory = ProfileMemory(
                    memory_type=MemoryType.CORE,
                    user_id=profile_user_id,
                    timestamp=get_now_with_timezone(),  # Profile通常没有具体时间戳
                    ori_event_id_list=[],
                    group_id=group_id,  # 设置为当前群组ID
                    participants=[profile_user_id],
                    subject="用户档案记忆",
                    summary=f"用户 {profile_user_id} 的档案信息",
                    episode=None,
                    type=RawDataType.CONVERSATION,  # Profile Memory 直接使用CONVERSATION类型
                    keywords=[],
                    linked_entities=[],
                    extend={
                        "core_memory_data": doc_profile
                    },  # 将原始CoreMemory数据存在extend中
                    user_name=doc_profile.user_name,
                    hard_skills=doc_profile.hard_skills,
                    soft_skills=doc_profile.soft_skills,
                    output_reasoning=getattr(doc_profile, "output_reasoning", None),
                    motivation_system=getattr(doc_profile, "motivation_system", None),
                    fear_system=getattr(doc_profile, "fear_system", None),
                    value_system=getattr(doc_profile, "value_system", None),
                    humor_use=getattr(doc_profile, "humor_use", None),
                    colloquialism=getattr(doc_profile, "colloquialism", None),
                    personality=doc_profile.personality,
                    way_of_decision_making=doc_profile.way_of_decision_making,
                    projects_participated=_convert_projects_participated_list(
                        doc_profile.projects_participated
                    ),
                    user_goal=doc_profile.user_goal,
                    work_responsibility=doc_profile.work_responsibility,
                    working_habit_preference=doc_profile.working_habit_preference,
                    interests=doc_profile.interests,
                    tendency=doc_profile.tendency,
                )
                profile_memory.memory_type = MemoryType.CORE
                profiles_by_group[group_id].append(profile_memory)

    # 统计分组结果
    all_groups = set(
        list(memcells_by_group.keys())
        + list(episodes_by_group.keys())
        + list(profiles_by_group.keys())
    )
    logger.info(f"[memorize_offline] 数据分组完成，共发现 {len(all_groups)} 个群组:")
    for group_id in all_groups:
        memcell_count = len(memcells_by_group.get(group_id, []))
        episode_count = len(episodes_by_group.get(group_id, []))
        profile_count = len(profiles_by_group.get(group_id, []))
        logger.info(
            f"  - {group_id}: MemCell={memcell_count}, Episode={episode_count}, Profile={profile_count}"
        )

    # 检查是否有数据需要处理
    if not all_groups:
        logger.info(f"[memorize_offline] 没有发现任何群组数据，结束处理")
        return []

    # ========== 并发控制参数配置 ==========
    # 群组批次大小：控制每批次处理多少个群组，避免一次性创建过多任务（GROUP_BATCH_SIZE大于2*LLM_CONCURRENCY_LIMIT的时候，仅仅影响内存占用的大小）
    GROUP_BATCH_SIZE = 50

    # LLM调用并发度：统一控制所有LLM API调用的并发数（包括阶段1和阶段2，控制实际并发，实际并发为LLM_CONCURRENCY_LIMIT的两倍）
    # 阶段1：群组记忆提取时的LLM调用
    # 阶段2：用户Profile合并时的LLM调用
    LLM_CONCURRENCY_LIMIT = 25
    # ====================================

    def _batched(items: List[str], size: int) -> List[List[str]]:
        # 简单分批工具，避免一次创建过多任务
        return [items[i : i + size] for i in range(0, len(items), size)]

    all_group_ids = list(all_groups)
    version = get_version_from_request(request)

    all_extracted_memories = []

    # 创建全局LLM并发控制的Semaphore
    _global_llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)

    # 定义带LLM并发限制的群组处理函数
    async def memorize_offline_by_group_id_with_limit(
        group_id: str, memcells: List[MemCell], old_memory_list: List[Memory]
    ) -> List[Memory]:
        async with _global_llm_semaphore:
            return await memorize_offline_by_group_id(
                group_id, memcells, old_memory_list
            )

    # 阶段1: 按批次处理群组，提取 GROUP_PROFILE 和 PROFILE 记忆
    # 批次大小由 GROUP_BATCH_SIZE 控制，LLM并发由 Semaphore 控制
    logger.info(
        f"[memorize_offline] 开始阶段1: 群组记忆提取，共 {len(all_group_ids)} 个群组，批次大小 {GROUP_BATCH_SIZE}，LLM并发度 {LLM_CONCURRENCY_LIMIT}"
    )
    for group_batch in _batched(all_group_ids, GROUP_BATCH_SIZE):
        group_tasks = []
        task_names = []
        for group_id in group_batch:
            memcells = memcells_by_group.get(group_id, [])
            episodes = episodes_by_group.get(group_id, [])
            core_memory_list = profiles_by_group.get(group_id, [])

            # 读取群组用户画像（逐组读取即可，批次粒度已经控制住内存）
            doc_group_user_profiles = (
                await group_user_profile_memory_repo.get_by_group_id(group_id)
            )
            group_memory_list = []
            for doc_profile in doc_group_user_profiles:
                group_profile = ProfileMemory(
                    memory_type=MemoryType.PROFILE,
                    user_id=doc_profile.user_id,
                    ori_event_id_list=[],
                    timestamp=get_now_with_timezone(),
                    participants=[doc_profile.user_id],
                    subject="用户群组档案记忆",
                    summary=f"用户 {doc_profile.user_id} 在群 {group_id} 的档案信息",
                    episode=None,
                    type=RawDataType.CONVERSATION,
                    group_id=doc_profile.group_id,
                    user_name=doc_profile.user_name,
                    hard_skills=doc_profile.hard_skills,
                    soft_skills=doc_profile.soft_skills,
                    output_reasoning=getattr(doc_profile, "output_reasoning", None),
                    motivation_system=getattr(doc_profile, "motivation_system", None),
                    fear_system=getattr(doc_profile, "fear_system", None),
                    value_system=getattr(doc_profile, "value_system", None),
                    humor_use=getattr(doc_profile, "humor_use", None),
                    colloquialism=getattr(doc_profile, "colloquialism", None),
                    personality=doc_profile.personality,
                    way_of_decision_making=doc_profile.way_of_decision_making,
                    projects_participated=_convert_projects_participated_list(
                        doc_profile.projects_participated
                    ),
                    user_goal=doc_profile.user_goal,
                    work_responsibility=doc_profile.work_responsibility,
                    working_habit_preference=doc_profile.working_habit_preference,
                    interests=doc_profile.interests,
                    tendency=doc_profile.tendency,
                    group_importance_evidence=_convert_document_to_group_importance_evidence(
                        doc_profile.group_importance_evidence
                    ),
                )
                group_memory_list.append(group_profile)

            # 读取群组画像原始数据
            doc_group_profile_raw = await group_profile_raw_repo.get_by_group_id(
                group_id
            )
            user_group_memory_list = []
            if doc_group_profile_raw:
                converted_fields = _convert_group_profile_raw_to_memory_format(
                    doc_group_profile_raw
                )
                group_profile_raw = GroupProfileMemory(
                    memory_type=MemoryType.GROUP_PROFILE,
                    user_id="",
                    ori_event_id_list=[],
                    timestamp=converted_fields['timestamp'],
                    participants=[],
                    subject=(
                        doc_group_profile_raw.subject
                        if hasattr(doc_group_profile_raw, 'subject')
                        else "用户群组档案记忆"
                    ),
                    summary=(
                        doc_group_profile_raw.summary
                        if hasattr(doc_group_profile_raw, 'summary')
                        else f"用户在群 {group_id} 的档案信息"
                    ),
                    group_id=doc_group_profile_raw.group_id,
                    group_name=doc_group_profile_raw.group_name,
                    topics=converted_fields['topics'],
                    roles=converted_fields['roles'],
                    extend=converted_fields['extend'],
                )
                user_group_memory_list.append(group_profile_raw)

            old_memory_list = (
                episodes + core_memory_list + group_memory_list + user_group_memory_list
            )

            if memcells:
                # 使用带LLM并发限制的函数
                task = asyncio.create_task(
                    memorize_offline_by_group_id_with_limit(
                        group_id, memcells, old_memory_list
                    ),
                    name=f"process_group_{group_id}",
                )
                group_tasks.append(task)
                task_names.append(task.get_name())
                logger.info(
                    f"[memorize_offline] 创建群组任务: {group_id}, MemCell={len(memcells)}, 历史记忆={len(old_memory_list)}"
                )
            else:
                logger.info(
                    f"[memorize_offline] 群组 {group_id} 没有MemCell数据，跳过处理"
                )

        if not group_tasks:
            logger.info(f"[memorize_offline] 本批次无可处理的群组任务")
            continue

        logger.info(f"[memorize_offline] 并行处理本批次 {len(group_tasks)} 个群组...")

        # 执行本批次任务
        group_results = await asyncio.gather(*group_tasks, return_exceptions=True)

        # 整理与落库（就地保存释放内存）
        batch_memories: List[Memory] = []

        # 整理与落库（就地保存释放内存）
        batch_memories: List[Memory] = []
        for i, result in enumerate(group_results):
            task_name = task_names[i] if i < len(task_names) else "unknown_task"
            task_name = task_names[i] if i < len(task_names) else "unknown_task"
            if isinstance(result, Exception):
                logger.error(
                    f"[memorize_offline] 群组任务 {task_name} 执行失败: {result}"
                )
            elif isinstance(result, list):
                batch_memories.extend(result)
                batch_memories.extend(result)
            else:
                logger.warning(
                    f"[memorize_offline] 群组任务 {task_name} 返回未知格式: {type(result)}"
                )

        if batch_memories:
            logger.info(
                f"[memorize_offline] 本批次提取 {len(batch_memories)} 个新记忆，准备保存..."
            )
            try:
                await save_memories(batch_memories, current_time, version)
                logger.info(f"[memorize_offline] 本批次记忆保存成功")
            except Exception as e:
                logger.error(f"[memorize_offline] 保存本批次记忆失败: {e}")

            # 汇总到总结果（保持函数返回行为）
            all_extracted_memories.extend(batch_memories)

    # 阶段2: LLM合并用户profile记忆（跨群组合并同一用户的所有profile）
    # 使用同一个Semaphore控制LLM并发度，避免超出API限制
    if all_extracted_memories:
        profile_memories = [
            m for m in all_extracted_memories if m.memory_type == MemoryType.PROFILE
        ]
        user_ids = [m.user_id for m in profile_memories if m.user_id]
        user_ids = list(set(user_ids))

        if user_ids:
            # 构建 core memory 的 user_id -> ProfileMemory 映射
            # 复用已经读取的 doc_profiles 数据
            core_profile_map: Dict[str, ProfileMemory] = {}
            for doc_profile in doc_profiles:
                if doc_profile and doc_profile.user_id in user_ids:
                    # 使用与 864-905 行相同的转换逻辑
                    core_profile_map[doc_profile.user_id] = ProfileMemory(
                        memory_type=MemoryType.CORE,
                        user_id=doc_profile.user_id,
                        timestamp=get_now_with_timezone(),
                        ori_event_id_list=[],
                        group_id=None,  # 用户级别的 core memory 不特定于群组
                        participants=[doc_profile.user_id],
                        subject="用户档案记忆",
                        summary=f"用户 {doc_profile.user_id} 的档案信息",
                        episode=None,
                        type=RawDataType.CONVERSATION,
                        keywords=[],
                        linked_entities=[],
                        extend={"core_memory_data": doc_profile},
                        user_name=doc_profile.user_name,
                        hard_skills=doc_profile.hard_skills,
                        soft_skills=doc_profile.soft_skills,
                        output_reasoning=getattr(doc_profile, "output_reasoning", None),
                        motivation_system=getattr(
                            doc_profile, "motivation_system", None
                        ),
                        fear_system=getattr(doc_profile, "fear_system", None),
                        value_system=getattr(doc_profile, "value_system", None),
                        humor_use=getattr(doc_profile, "humor_use", None),
                        colloquialism=getattr(doc_profile, "colloquialism", None),
                        personality=doc_profile.personality,
                        way_of_decision_making=doc_profile.way_of_decision_making,
                        projects_participated=_convert_projects_participated_list(
                            doc_profile.projects_participated
                        ),
                        user_goal=doc_profile.user_goal,
                        work_responsibility=doc_profile.work_responsibility,
                        working_habit_preference=doc_profile.working_habit_preference,
                        interests=doc_profile.interests,
                        tendency=doc_profile.tendency,
                    )

            logger.info(
                f"[memorize_offline] 准备合并 {len(user_ids)} 个用户的 profile，其中 {len(core_profile_map)} 个有 core memory"
            )

            profile_merger = ProfileMemoryMerger(
                memory_manager.profile_memory_extractor_llm_provider
            )

            # 定义带LLM并发限制的用户profile合并函数（使用全局Semaphore）
            async def merge_profile_for_user_with_limit(user_id: str):
                """带并发限制的用户profile合并函数（使用全局LLM Semaphore）"""
                async with _global_llm_semaphore:
                    group_profile_memories = [
                        m for m in profile_memories if m.user_id == user_id
                    ]
                    # 如果该用户有 core memory，也加入合并
                    if user_id in core_profile_map:
                        group_profile_memories.append(core_profile_map[user_id])
                    return await profile_merger.merge_group_profiles(
                        group_profile_memories, user_id
                    )

            # 按批次处理用户，避免一次性创建过多任务占用内存
            user_batch_size = LLM_CONCURRENCY_LIMIT * 2
            valid_user_profiles_total = []

            for batch_idx, user_batch in enumerate(
                _batched(user_ids, user_batch_size), 1
            ):
                logger.info(
                    f"[memorize_offline] 处理用户批次 {batch_idx}/{(len(user_ids) + user_batch_size - 1) // user_batch_size}，本批次 {len(user_batch)} 个用户"
                )

                # 并发执行本批次的用户profile合并（实际并发受Semaphore限制）
                user_results = await asyncio.gather(
                    *[merge_profile_for_user_with_limit(uid) for uid in user_batch],
                    return_exceptions=True,
                )

                # 收集成功的结果
                valid_user_profiles = []
                for i, result in enumerate(user_results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"[memorize_offline] 用户 {user_batch[i]} 的profile合并失败: {result}"
                        )
                    else:
                        valid_user_profiles.append(result)

                # 保存本批次合并后的profile
                if valid_user_profiles:
                    try:
                        await save_personal_profile_memory(valid_user_profiles, version)
                        logger.info(
                            f"[memorize_offline] 本批次 {len(valid_user_profiles)} 个用户profile保存成功"
                        )
                        valid_user_profiles_total.extend(valid_user_profiles)
                    except Exception as e:
                        logger.error(
                            f"[memorize_offline] 保存本批次用户profile失败: {e}"
                        )

            logger.info(
                f"[memorize_offline] 阶段2完成，共合并并保存 {len(valid_user_profiles_total)} 个用户profile"
            )

    logger.info(
        f"[memorize_offline] 离线记忆处理完成，共处理 {len(all_groups)} 个群组，提取 {len(all_extracted_memories)} 个新记忆"
    )
    return all_extracted_memories
