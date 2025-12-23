# -*- coding: utf-8 -*-
"""
ConversationDataRepository interface and implementation

Conversation data storage based on MemoryRequestLog, replacing the original Redis implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from core.observation.logger import get_logger
from core.di.decorators import repository
from core.di import get_bean
from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from biz_layer.mem_db_operations import _normalize_datetime_for_storage
from infra_layer.adapters.out.persistence.repository.memory_request_log_repository import (
    MemoryRequestLogRepository,
)
from infra_layer.adapters.out.persistence.mapper.memory_request_log_mapper import (
    MemoryRequestLogMapper,
)

logger = get_logger(__name__)


# ==================== Interface Definition ====================


class ConversationDataRepository(ABC):
    """Conversation data access interface"""

    @abstractmethod
    async def save_conversation_data(
        self,
        raw_data_list: List[RawData],
        group_id: str,
        auto_confirm_pending: bool = True,
    ) -> bool:
        """
        Save conversation data

        Args:
            raw_data_list: List of RawData to save
            group_id: Group ID
            auto_confirm_pending: If True (default), also update all sync_status=-1
                                  records to 0 for this group to handle edge cases

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_conversation_data(
        self,
        group_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        include_pending: bool = True,
    ) -> List[RawData]:
        """
        Get conversation data

        Args:
            group_id: Group ID
            start_time: Start time (ISO format string)
            end_time: End time (ISO format string)
            limit: Maximum number of records to return
            include_pending: If True (default), also include sync_status=-1 records
                             along with sync_status=0 records to handle edge cases

        Returns:
            List[RawData]: List of conversation data
        """
        pass

    @abstractmethod
    async def delete_conversation_data(self, group_id: str) -> bool:
        """
        Delete all conversation data for the specified group

        Args:
            group_id: Group ID

        Returns:
            bool: Return True if deletion succeeds, False otherwise
        """
        pass

    @abstractmethod
    async def fetch_unprocessed_conversation_data(
        self, group_id: str, limit: int = 100
    ) -> List[RawData]:
        """
        Fetch unprocessed conversation data (sync_status=-1 or 0)

        Unlike get_conversation_data, this method does not have time range filters
        and returns results in ascending order (oldest first).

        Args:
            group_id: Group ID
            limit: Maximum number of records to return (in ascending order)

        Returns:
            List[RawData]: List of unprocessed conversation data
        """
        pass


# ==================== Implementation ====================


@repository("conversation_data_repo", primary=True)
class ConversationDataRepositoryImpl(ConversationDataRepository):
    """
    ConversationDataRepository implementation based on MemoryRequestLog

    Reuses MemoryRequestLog storage for conversation data, converting between RawData
    and MemoryRequestLog. Data is automatically saved to MemoryRequestLog through
    the RequestHistoryEvent listener.
    """

    def __init__(self):
        self._repo: Optional[MemoryRequestLogRepository] = None

    def _get_repo(self) -> MemoryRequestLogRepository:
        """Lazy load MemoryRequestLogRepository"""
        if self._repo is None:
            self._repo = get_bean("memory_request_log_repository")
        return self._repo

    # ==================== ConversationDataRepository Interface Implementation ====================

    async def save_conversation_data(
        self,
        raw_data_list: List[RawData],
        group_id: str,
        auto_confirm_pending: bool = True,
    ) -> bool:
        """
        Confirm conversation data enters window accumulation

        Updates sync_status = -1 (log record) to sync_status = 0 (in window accumulation).
        The data itself is automatically saved to MemoryRequestLog through the
        RequestHistoryEvent listener; this method confirms these data entries
        enter the window accumulation state.

        Update strategy:
        - If raw_data_list contains data_id (i.e., message_id), precisely update these records
        - Otherwise, fall back to updating all sync_status=-1 records by group_id

        sync_status state transitions:
        - -1: Just a log record (raw request just saved via listener)
        -  0: In window accumulation (confirmed via this method)
        -  1: Already fully used (marked via delete_conversation_data)

        Args:
            raw_data_list: RawData list, used to extract message_id for precise updates
            group_id: Conversation group ID
            auto_confirm_pending: If True (default), also confirm all pending (-1) records
                                  for this group to handle edge cases

        Returns:
            bool: True if operation succeeds, False otherwise
        """
        logger.info(
            "Confirming conversation data enters window accumulation: group_id=%s, data_count=%d",
            group_id,
            len(raw_data_list) if raw_data_list else 0,
        )

        try:
            repo = self._get_repo()

            # Extract message_id list (filter out empty values)
            message_ids = [r.data_id for r in (raw_data_list or []) if r.data_id]

            if message_ids:
                # Precise update: only update records with specified message_id
                modified_count = await repo.confirm_accumulation_by_message_ids(
                    group_id, message_ids
                )

                # If auto_confirm_pending is True, also confirm all remaining pending records
                if auto_confirm_pending:
                    extra_modified = await repo.confirm_accumulation_by_group_id(
                        group_id
                    )
                    if extra_modified > 0:
                        logger.info(
                            "Auto-confirmed additional pending records: group_id=%s, extra_modified=%d",
                            group_id,
                            extra_modified,
                        )
                        modified_count += extra_modified
            else:
                # Fallback: update all sync_status=-1 records by group_id
                logger.debug(
                    "No data_id in raw_data_list, falling back to update by group_id"
                )
                modified_count = await repo.confirm_accumulation_by_group_id(group_id)

            logger.info(
                "Window accumulation confirmation completed: group_id=%s, message_ids=%d, modified=%d",
                group_id,
                len(message_ids),
                modified_count,
            )
            return True

        except Exception as e:
            logger.error(
                "Window accumulation confirmation failed: group_id=%s, error=%s",
                group_id,
                e,
            )
            return False

    async def get_conversation_data(
        self,
        group_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        include_pending: bool = True,
    ) -> List[RawData]:
        """
        Get conversation data in window accumulation

        Queries MemoryRequestLog with sync_status = 0 (in window accumulation) and
        converts to RawData. Only returns data that has been confirmed to enter
        the accumulation window but has not yet been used.

        sync_status state description:
        - -1: Just a log record (not returned by default, unless include_pending=True)
        -  0: In window accumulation (returned)
        -  1: Already fully used (not returned)

        Args:
            group_id: Conversation group ID
            start_time: Start time (ISO format string)
            end_time: End time (ISO format string)
            limit: Maximum number of records to return
            include_pending: If True (default), also include sync_status=-1 records
                             along with sync_status=0 records to handle edge cases

        Returns:
            List[RawData]: List of conversation data
        """
        logger.info(
            "Fetching conversation data: group_id=%s, start_time=%s, end_time=%s, limit=%d, include_pending=%s",
            group_id,
            start_time,
            end_time,
            limit,
            include_pending,
        )

        try:
            repo = self._get_repo()

            # Convert time format
            start_dt = (
                _normalize_datetime_for_storage(start_time) if start_time else None
            )
            end_dt = _normalize_datetime_for_storage(end_time) if end_time else None

            # Determine which sync_status values to query
            if include_pending:
                # Include both pending (-1) and accumulating (0) records
                sync_status_list = [-1, 0]
            else:
                # Only include accumulating (0) records
                sync_status_list = [0]

            # Query MemoryRequestLog with multiple sync_status values
            logs = await repo.find_by_group_id_with_statuses(
                group_id=group_id,
                sync_status_list=sync_status_list,
                start_time=start_dt,
                end_time=end_dt,
                limit=limit,
            )

            # Use mapper to convert to RawData list
            raw_data_list = MemoryRequestLogMapper.to_raw_data_list(logs)

            logger.info(
                "Conversation data fetch completed: group_id=%s, count=%d",
                group_id,
                len(raw_data_list),
            )
            return raw_data_list

        except Exception as e:
            logger.error(
                "Conversation data fetch failed: group_id=%s, error=%s", group_id, e
            )
            return []

    async def delete_conversation_data(self, group_id: str) -> bool:
        """
        Mark accumulated data for the specified conversation group as used

        Updates sync_status = 0 (in window accumulation) to sync_status = 1 (fully used).
        Note: This method does not actually delete data, but updates the sync_status state.
        This preserves historical data for auditing and replay while not affecting
        subsequent queries.

        sync_status state transitions:
        - -1: Just a log record
        -  0: In window accumulation (confirmed via save_conversation_data)
        -  1: Already fully used (marked via this method, called after boundary detection)

        Args:
            group_id: Conversation group ID

        Returns:
            bool: True if operation succeeds, False otherwise
        """
        logger.info("Marking conversation data as used: group_id=%s", group_id)

        try:
            repo = self._get_repo()
            # Update sync_status: 0 -> 1
            modified_count = await repo.mark_as_used_by_group_id(group_id)

            logger.info(
                "Conversation data marked as used: group_id=%s, modified=%d",
                group_id,
                modified_count,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to mark conversation data as used: group_id=%s, error=%s",
                group_id,
                e,
            )
            return False

    async def fetch_unprocessed_conversation_data(
        self, group_id: str, limit: int = 100
    ) -> List[RawData]:
        """
        Fetch unprocessed conversation data (sync_status=-1 or 0)

        Unlike get_conversation_data, this method:
        - Does not have start_time and end_time filters
        - Returns results in ascending order (oldest first) with limit applied

        This is useful for fetching all pending/accumulating messages that need
        to be processed, without time range restrictions.

        Args:
            group_id: Conversation group ID
            limit: Maximum number of records to return (in ascending order, oldest first)

        Returns:
            List[RawData]: List of unprocessed conversation data
        """
        logger.info(
            "Fetching unprocessed conversation data: group_id=%s, limit=%d",
            group_id,
            limit,
        )

        try:
            repo = self._get_repo()

            # Query both pending (-1) and accumulating (0) records
            # No time range filter, ascending order (oldest first)
            logs = await repo.find_by_group_id_with_statuses(
                group_id=group_id,
                sync_status_list=[-1, 0],
                start_time=None,
                end_time=None,
                limit=limit,
                ascending=True,
            )

            # Use mapper to convert to RawData list
            raw_data_list = MemoryRequestLogMapper.to_raw_data_list(logs)

            logger.info(
                "Unprocessed conversation data fetch completed: group_id=%s, count=%d",
                group_id,
                len(raw_data_list),
            )
            return raw_data_list

        except Exception as e:
            logger.error(
                "Unprocessed conversation data fetch failed: group_id=%s, error=%s",
                group_id,
                e,
            )
            return []
