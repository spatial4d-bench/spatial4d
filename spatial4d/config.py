"""Benchmark task registry and compatibility constants (no I/O)."""
from dataclasses import dataclass
from types import MappingProxyType

RESULT_DIR = "/data1/wp/code/qwen/eval_results/Qwen/Qwen3.5-9B/"

# A structural refactor does not change scores or invalidate judge caches.
RULES_VERSION = "3.0-v3-optional-openai-judge"
PLATFORM_SOURCE_SHA256 = "1e5c742a6338fc51e3f061f91a2568b7a00a8819917367f6ecdea17279031a1e"
OPTION_LABELS = "ABCDEF"
FAIL_MSG = "Failed to obtain answer via API."


@dataclass(frozen=True)
class TaskSpec:
    name: str
    metric: str


# Keep the legacy MCA/NA ordering for callers; reports use sorted TASKS.
TASK_SPECS = MappingProxyType({spec.name: spec for spec in (
    TaskSpec("3d_grounding", "accuracy"),
    TaskSpec("action_prediction", "accuracy"),
    TaskSpec("action_recognition", "accuracy"),
    TaskSpec("affordance_det", "accuracy"),
    TaskSpec("app_order_reasoning", "accuracy"),
    TaskSpec("egocentric_reasoning", "accuracy"),
    TaskSpec("object_attr_est", "accuracy"),
    TaskSpec("phy_plau_reasoning", "accuracy"),
    TaskSpec("rel_ori_est", "accuracy"),
    TaskSpec("route_plan", "accuracy"),
    TaskSpec("scene_classification", "accuracy"),
    TaskSpec("spatial_memory", "accuracy"),
    TaskSpec("state_change_det", "accuracy"),
    TaskSpec("rel_dist_est", "accuracy"),
    TaskSpec("abs_dist_est", "MRA"),
    TaskSpec("object_counting", "MRA"),
    TaskSpec("object_size_est", "MRA"),
    TaskSpec("room_size_estimation", "MRA"),
)})
MCA_QUESTION_TYPES = [name for name, spec in TASK_SPECS.items() if spec.metric == "accuracy"]
NA_QUESTION_TYPES = [name for name, spec in TASK_SPECS.items() if spec.metric == "MRA"]
TASKS = sorted(TASK_SPECS)
ALIASES = {
    "absolute_distance_estimation": "abs_dist_est", "affordance_detection": "affordance_det",
    "appearance_order_reasoning": "app_order_reasoning", "object_attribute_estimation": "object_attr_est",
    "object_size_estimation": "object_size_est", "dynamic_physics": "phy_plau_reasoning",
    "relative_distance_estimation": "rel_dist_est", "relative_orientation_estimation": "rel_ori_est",
    "state_change_detection": "state_change_det",
}
