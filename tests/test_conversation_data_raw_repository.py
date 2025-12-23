#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the functionality of ConversationDataRepository

Test contents include:
1. save_conversation_data with auto_confirm_pending parameter
2. get_conversation_data with include_pending parameter
3. delete_conversation_data
4. fetch_unprocessed_conversation_data
5. sync_status state transitions
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List

from core.di import get_bean_by_type
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from infra_layer.adapters.out.persistence.repository.conversation_data_raw_repository import (
    ConversationDataRepository,
)
from infra_layer.adapters.out.persistence.repository.memory_request_log_repository import (
    MemoryRequestLogRepository,
)
from infra_layer.adapters.out.persistence.document.request.memory_request_log import (
    MemoryRequestLog,
)
from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from core.observation.logger import get_logger

logger = get_logger(__name__)


def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique ID for testing"""
    return f"{prefix}{uuid.uuid4().hex[:8]}"


async def create_test_memory_request_log(
    group_id: str,
    message_id: str,
    content: str,
    sync_status: int = -1,
    created_at: datetime = None,
) -> MemoryRequestLog:
    """
    Create a test MemoryRequestLog record

    Args:
        group_id: Group ID
        message_id: Message ID
        content: Message content
        sync_status: Sync status (-1=log, 0=accumulating, 1=used)
        created_at: Created time (default: now)

    Returns:
        Created MemoryRequestLog object
    """
    log_repo = get_bean_by_type(MemoryRequestLogRepository)

    log = MemoryRequestLog(
        group_id=group_id,
        request_id=generate_unique_id("req_"),
        message_id=message_id,
        sender="test_user",
        sender_name="Test User",
        content=content,
        message_create_time=to_iso_format(created_at or get_now_with_timezone()),
        sync_status=sync_status,
    )

    # Manually set created_at if provided
    if created_at:
        log.created_at = created_at

    await log_repo.save(log)
    return log


async def cleanup_test_data(group_id: str):
    """Clean up test data for a group"""
    log_repo = get_bean_by_type(MemoryRequestLogRepository)
    await log_repo.delete_by_group_id(group_id)


async def get_logs_by_group_id(group_id: str) -> List[MemoryRequestLog]:
    """Get all logs for a group (for verification)"""
    log_repo = get_bean_by_type(MemoryRequestLogRepository)
    return await log_repo.find_by_group_id(group_id, sync_status=None)


async def test_save_conversation_data_basic():
    """Test basic save_conversation_data functionality"""
    logger.info("Starting test for basic save_conversation_data...")

    repo = get_bean_by_type(ConversationDataRepository)
    group_id = generate_unique_id("test_save_basic_")

    try:
        # Create test log records with sync_status=-1
        msg1 = await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Test message 1",
            sync_status=-1,
        )
        msg2 = await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Test message 2",
            sync_status=-1,
        )
        logger.info("✅ Created 2 test log records with sync_status=-1")

        # Create RawData list with data_id
        raw_data_list = [
            RawData(
                data_id=msg1.message_id,
                content={"content": "Test message 1"},
                data_type="message",
            ),
            RawData(
                data_id=msg2.message_id,
                content={"content": "Test message 2"},
                data_type="message",
            ),
        ]

        # Save conversation data
        result = await repo.save_conversation_data(raw_data_list, group_id)
        assert result is True
        logger.info("✅ save_conversation_data returned True")

        # Verify sync_status changed to 0
        logs = await get_logs_by_group_id(group_id)
        for log in logs:
            assert (
                log.sync_status == 0
            ), f"Expected sync_status=0, got {log.sync_status}"
        logger.info("✅ All records now have sync_status=0")

    except Exception as e:
        logger.error("❌ Test for basic save_conversation_data failed: %s", e)
        raise
    finally:
        await cleanup_test_data(group_id)
        logger.info("✅ Cleaned up test data")

    logger.info("✅ Basic save_conversation_data test completed")


async def test_save_conversation_data_auto_confirm_pending():
    """Test save_conversation_data with auto_confirm_pending parameter"""
    logger.info("Starting test for save_conversation_data with auto_confirm_pending...")

    repo = get_bean_by_type(ConversationDataRepository)
    group_id = generate_unique_id("test_auto_confirm_")

    try:
        # Create 3 log records with sync_status=-1
        msg1_id = generate_unique_id("msg_")
        msg2_id = generate_unique_id("msg_")
        msg3_id = generate_unique_id("msg_")

        await create_test_memory_request_log(
            group_id=group_id, message_id=msg1_id, content="Message 1", sync_status=-1
        )
        await create_test_memory_request_log(
            group_id=group_id, message_id=msg2_id, content="Message 2", sync_status=-1
        )
        await create_test_memory_request_log(
            group_id=group_id, message_id=msg3_id, content="Message 3", sync_status=-1
        )
        logger.info("✅ Created 3 test log records with sync_status=-1")

        # Save only msg1 with auto_confirm_pending=True (default)
        # This should confirm msg1 precisely AND also confirm msg2, msg3 automatically
        raw_data_list = [
            RawData(
                data_id=msg1_id, content={"content": "Message 1"}, data_type="message"
            )
        ]

        result = await repo.save_conversation_data(
            raw_data_list, group_id, auto_confirm_pending=True
        )
        assert result is True
        logger.info(
            "✅ save_conversation_data with auto_confirm_pending=True succeeded"
        )

        # Verify all 3 records have sync_status=0
        logs = await get_logs_by_group_id(group_id)
        assert len(logs) == 3
        for log in logs:
            assert (
                log.sync_status == 0
            ), f"Expected sync_status=0, got {log.sync_status}"
        logger.info("✅ All 3 records have sync_status=0 (auto-confirmed)")

    except Exception as e:
        logger.error("❌ Test for auto_confirm_pending failed: %s", e)
        raise
    finally:
        await cleanup_test_data(group_id)
        logger.info("✅ Cleaned up test data")

    logger.info("✅ auto_confirm_pending test completed")


async def test_save_conversation_data_auto_confirm_pending_false():
    """Test save_conversation_data with auto_confirm_pending=False"""
    logger.info(
        "Starting test for save_conversation_data with auto_confirm_pending=False..."
    )

    repo = get_bean_by_type(ConversationDataRepository)
    group_id = generate_unique_id("test_no_auto_confirm_")

    try:
        # Create 3 log records with sync_status=-1
        msg1_id = generate_unique_id("msg_")
        msg2_id = generate_unique_id("msg_")
        msg3_id = generate_unique_id("msg_")

        await create_test_memory_request_log(
            group_id=group_id, message_id=msg1_id, content="Message 1", sync_status=-1
        )
        await create_test_memory_request_log(
            group_id=group_id, message_id=msg2_id, content="Message 2", sync_status=-1
        )
        await create_test_memory_request_log(
            group_id=group_id, message_id=msg3_id, content="Message 3", sync_status=-1
        )
        logger.info("✅ Created 3 test log records with sync_status=-1")

        # Save only msg1 with auto_confirm_pending=False
        # This should only confirm msg1, leaving msg2 and msg3 with sync_status=-1
        raw_data_list = [
            RawData(
                data_id=msg1_id, content={"content": "Message 1"}, data_type="message"
            )
        ]

        result = await repo.save_conversation_data(
            raw_data_list, group_id, auto_confirm_pending=False
        )
        assert result is True
        logger.info(
            "✅ save_conversation_data with auto_confirm_pending=False succeeded"
        )

        # Verify only msg1 has sync_status=0
        logs = await get_logs_by_group_id(group_id)
        assert len(logs) == 3

        confirmed_count = 0
        pending_count = 0
        for log in logs:
            if log.message_id == msg1_id:
                assert log.sync_status == 0, "msg1 should have sync_status=0"
                confirmed_count += 1
            else:
                assert log.sync_status == -1, "Other msgs should have sync_status=-1"
                pending_count += 1

        assert confirmed_count == 1
        assert pending_count == 2
        logger.info("✅ Only msg1 has sync_status=0, others remain at -1")

    except Exception as e:
        logger.error("❌ Test for auto_confirm_pending=False failed: %s", e)
        raise
    finally:
        await cleanup_test_data(group_id)
        logger.info("✅ Cleaned up test data")

    logger.info("✅ auto_confirm_pending=False test completed")


async def test_get_conversation_data_include_pending():
    """Test get_conversation_data with include_pending parameter"""
    logger.info("Starting test for get_conversation_data with include_pending...")

    repo = get_bean_by_type(ConversationDataRepository)
    group_id = generate_unique_id("test_include_pending_")

    try:
        # Create logs with different sync_status
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Pending message",
            sync_status=-1,  # pending
        )
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Accumulating message",
            sync_status=0,  # accumulating
        )
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Used message",
            sync_status=1,  # used
        )
        logger.info("✅ Created logs with sync_status -1, 0, 1")

        # Test include_pending=True (default) - should get both -1 and 0
        result_with_pending = await repo.get_conversation_data(
            group_id=group_id, include_pending=True
        )
        assert (
            len(result_with_pending) == 2
        ), f"Expected 2 results, got {len(result_with_pending)}"
        logger.info("✅ include_pending=True returned 2 records (-1 and 0)")

        # Test include_pending=False - should only get 0
        result_without_pending = await repo.get_conversation_data(
            group_id=group_id, include_pending=False
        )
        assert (
            len(result_without_pending) == 1
        ), f"Expected 1 result, got {len(result_without_pending)}"
        logger.info("✅ include_pending=False returned 1 record (only 0)")

    except Exception as e:
        logger.error("❌ Test for include_pending failed: %s", e)
        raise
    finally:
        await cleanup_test_data(group_id)
        logger.info("✅ Cleaned up test data")

    logger.info("✅ include_pending test completed")


async def test_delete_conversation_data():
    """Test delete_conversation_data (marks as used)"""
    logger.info("Starting test for delete_conversation_data...")

    repo = get_bean_by_type(ConversationDataRepository)
    group_id = generate_unique_id("test_delete_")

    try:
        # Create logs with sync_status=-1 and 0
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Pending message",
            sync_status=-1,
        )
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Accumulating message",
            sync_status=0,
        )
        logger.info("✅ Created logs with sync_status -1 and 0")

        # Delete (mark as used)
        result = await repo.delete_conversation_data(group_id)
        assert result is True
        logger.info("✅ delete_conversation_data returned True")

        # Verify all records now have sync_status=1
        logs = await get_logs_by_group_id(group_id)
        assert len(logs) == 2
        for log in logs:
            assert (
                log.sync_status == 1
            ), f"Expected sync_status=1, got {log.sync_status}"
        logger.info("✅ All records now have sync_status=1 (marked as used)")

        # Verify get_conversation_data returns empty
        remaining = await repo.get_conversation_data(group_id)
        assert len(remaining) == 0, f"Expected 0 results, got {len(remaining)}"
        logger.info("✅ get_conversation_data returns empty after deletion")

    except Exception as e:
        logger.error("❌ Test for delete_conversation_data failed: %s", e)
        raise
    finally:
        await cleanup_test_data(group_id)
        logger.info("✅ Cleaned up test data")

    logger.info("✅ delete_conversation_data test completed")


async def test_fetch_unprocessed_conversation_data():
    """Test fetch_unprocessed_conversation_data"""
    logger.info("Starting test for fetch_unprocessed_conversation_data...")

    repo = get_bean_by_type(ConversationDataRepository)
    group_id = generate_unique_id("test_fetch_unprocessed_")

    try:
        now = get_now_with_timezone()

        # Create logs with different sync_status and timestamps
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Oldest pending",
            sync_status=-1,
            created_at=now - timedelta(hours=3),
        )
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Middle accumulating",
            sync_status=0,
            created_at=now - timedelta(hours=2),
        )
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Newest pending",
            sync_status=-1,
            created_at=now - timedelta(hours=1),
        )
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Used message",
            sync_status=1,
            created_at=now,
        )
        logger.info("✅ Created 4 logs with different sync_status and timestamps")

        # Fetch unprocessed data
        result = await repo.fetch_unprocessed_conversation_data(group_id, limit=100)

        # Should get 3 records (sync_status=-1 and 0, excluding sync_status=1)
        assert len(result) == 3, f"Expected 3 results, got {len(result)}"
        logger.info("✅ fetch_unprocessed returned 3 records")

        # Verify ascending order (oldest first)
        # RawData.content is a dict, and the content field contains the message text
        assert "Oldest pending" in str(result[0].content.get("content", ""))
        assert "Middle accumulating" in str(result[1].content.get("content", ""))
        assert "Newest pending" in str(result[2].content.get("content", ""))
        logger.info("✅ Results are in ascending order (oldest first)")

        # Test with limit
        limited_result = await repo.fetch_unprocessed_conversation_data(
            group_id, limit=2
        )
        assert (
            len(limited_result) == 2
        ), f"Expected 2 results with limit, got {len(limited_result)}"
        assert "Oldest pending" in str(limited_result[0].content.get("content", ""))
        assert "Middle accumulating" in str(
            limited_result[1].content.get("content", "")
        )
        logger.info("✅ Limit parameter works correctly")

    except Exception as e:
        logger.error("❌ Test for fetch_unprocessed_conversation_data failed: %s", e)
        raise
    finally:
        await cleanup_test_data(group_id)
        logger.info("✅ Cleaned up test data")

    logger.info("✅ fetch_unprocessed_conversation_data test completed")


async def test_sync_status_state_transitions():
    """Test the complete sync_status state transition flow"""
    logger.info("Starting test for sync_status state transitions...")

    repo = get_bean_by_type(ConversationDataRepository)
    group_id = generate_unique_id("test_state_flow_")

    try:
        # Step 1: Create initial log (simulating RequestHistoryEvent listener)
        # New logs start with sync_status=-1
        msg_id = generate_unique_id("msg_")
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=msg_id,
            content="Test message for state flow",
            sync_status=-1,
        )
        logger.info("✅ Step 1: Created log with sync_status=-1 (log record)")

        # Verify initial state
        logs = await get_logs_by_group_id(group_id)
        assert len(logs) == 1
        assert logs[0].sync_status == -1
        logger.info("✅ Verified initial sync_status=-1")

        # Step 2: save_conversation_data -> sync_status becomes 0
        raw_data_list = [
            RawData(
                data_id=msg_id, content={"content": "Test message"}, data_type="message"
            )
        ]
        await repo.save_conversation_data(raw_data_list, group_id)

        logs = await get_logs_by_group_id(group_id)
        assert logs[0].sync_status == 0
        logger.info("✅ Step 2: sync_status changed to 0 (window accumulation)")

        # Step 3: delete_conversation_data -> sync_status becomes 1
        await repo.delete_conversation_data(group_id)

        logs = await get_logs_by_group_id(group_id)
        assert logs[0].sync_status == 1
        logger.info("✅ Step 3: sync_status changed to 1 (used)")

        # Verify the message is no longer retrievable
        result = await repo.get_conversation_data(group_id)
        assert len(result) == 0
        logger.info("✅ Verified: used messages are not retrievable")

        result_unprocessed = await repo.fetch_unprocessed_conversation_data(group_id)
        assert len(result_unprocessed) == 0
        logger.info("✅ Verified: used messages not in unprocessed")

    except Exception as e:
        logger.error("❌ Test for sync_status state transitions failed: %s", e)
        raise
    finally:
        await cleanup_test_data(group_id)
        logger.info("✅ Cleaned up test data")

    logger.info("✅ sync_status state transitions test completed")


async def test_empty_raw_data_list():
    """Test save_conversation_data with empty raw_data_list"""
    logger.info("Starting test for empty raw_data_list...")

    repo = get_bean_by_type(ConversationDataRepository)
    group_id = generate_unique_id("test_empty_list_")

    try:
        # Create log records
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Message 1",
            sync_status=-1,
        )
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Message 2",
            sync_status=-1,
        )
        logger.info("✅ Created 2 log records")

        # Save with empty list - should fall back to group_id update
        result = await repo.save_conversation_data([], group_id)
        assert result is True
        logger.info("✅ save_conversation_data with empty list returned True")

        # Verify all records are confirmed (fallback behavior)
        logs = await get_logs_by_group_id(group_id)
        for log in logs:
            assert (
                log.sync_status == 0
            ), f"Expected sync_status=0, got {log.sync_status}"
        logger.info("✅ All records confirmed via fallback (empty list)")

    except Exception as e:
        logger.error("❌ Test for empty raw_data_list failed: %s", e)
        raise
    finally:
        await cleanup_test_data(group_id)
        logger.info("✅ Cleaned up test data")

    logger.info("✅ empty raw_data_list test completed")


async def test_raw_data_list_without_data_id():
    """Test save_conversation_data with raw_data_list that has no data_id"""
    logger.info("Starting test for raw_data_list without data_id...")

    repo = get_bean_by_type(ConversationDataRepository)
    group_id = generate_unique_id("test_no_data_id_")

    try:
        # Create log records
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Message 1",
            sync_status=-1,
        )
        await create_test_memory_request_log(
            group_id=group_id,
            message_id=generate_unique_id("msg_"),
            content="Message 2",
            sync_status=-1,
        )
        logger.info("✅ Created 2 log records")

        # Save with RawData that has no data_id
        raw_data_list = [
            RawData(
                data_id="",
                content={"content": "Content without ID"},
                data_type="message",
            )
        ]

        result = await repo.save_conversation_data(raw_data_list, group_id)
        assert result is True
        logger.info("✅ save_conversation_data with no data_id returned True")

        # Verify all records are confirmed (fallback behavior)
        logs = await get_logs_by_group_id(group_id)
        for log in logs:
            assert (
                log.sync_status == 0
            ), f"Expected sync_status=0, got {log.sync_status}"
        logger.info("✅ All records confirmed via fallback (no data_id)")

    except Exception as e:
        logger.error("❌ Test for raw_data_list without data_id failed: %s", e)
        raise
    finally:
        await cleanup_test_data(group_id)
        logger.info("✅ Cleaned up test data")

    logger.info("✅ raw_data_list without data_id test completed")


async def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting to run all ConversationDataRepository tests...")

    try:
        await test_save_conversation_data_basic()
        await test_save_conversation_data_auto_confirm_pending()
        await test_save_conversation_data_auto_confirm_pending_false()
        await test_get_conversation_data_include_pending()
        await test_delete_conversation_data()
        await test_fetch_unprocessed_conversation_data()
        await test_sync_status_state_transitions()
        await test_empty_raw_data_list()
        await test_raw_data_list_without_data_id()
        logger.info("✅ All tests completed successfully")
    except Exception as e:
        logger.error("❌ Error occurred during testing: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
