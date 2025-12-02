"""
记忆提取流程配置

所有触发条件和阈值集中管理，便于调整和维护。
"""

from dataclasses import dataclass
import os


@dataclass
class MemorizeConfig:
    """记忆提取流程配置"""

    # ===== 聚类配置 =====
    # 语义相似度阈值，超过此值的 memcells 会被聚到同一个 cluster
    cluster_similarity_threshold: float = 0.3
    # 最大时间间隔（天），超过此间隔的 memcells 不会被聚到一起
    cluster_max_time_gap_days: int = 7

    # ===== Profile 提取配置 =====
    # 触发 Profile 提取的最小 memcells 数量
    profile_min_memcells: int = 1
    # Profile 提取的最小置信度
    profile_min_confidence: float = 0.6
    # 是否启用版本控制
    profile_enable_versioning: bool = True

    @classmethod
    def from_env(cls) -> "MemorizeConfig":
        """从环境变量加载配置，未设置则使用默认值"""
        return cls(
            cluster_similarity_threshold=float(
                os.getenv("CLUSTER_SIMILARITY_THRESHOLD", "0.3")
            ),
            cluster_max_time_gap_days=int(
                os.getenv("CLUSTER_MAX_TIME_GAP_DAYS", "7")
            ),
            profile_min_memcells=int(
                os.getenv("PROFILE_MIN_MEMCELLS", "1")
            ),
            profile_min_confidence=float(
                os.getenv("PROFILE_MIN_CONFIDENCE", "0.6")
            ),
            profile_enable_versioning=os.getenv(
                "PROFILE_ENABLE_VERSIONING", "true"
            ).lower() == "true",
        )


# 全局默认配置（可通过 from_env() 覆盖）
DEFAULT_MEMORIZE_CONFIG = MemorizeConfig()

