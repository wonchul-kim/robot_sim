# source/isaaclab_assets/utils/attachments.py
from typing import Dict, Any
import isaaclab.sim as sim_utils

def attach_articulation_fixed(env, params: Dict[str, Any]):
    """Attach 'child_asset' articulation under 'parent_asset' link with fixed joint."""
    parent_asset = params.get("parent_asset", "robot")
    child_asset  = params.get("child_asset", "gripper")
    parent_link  = params.get("parent_link", "tool0")
    pose_dict    = params.get("pose_in_parent", dict(p=[0,0,0], rpy=[0,0,0]))
    fixed_joint  = params.get("fixed_joint", True)

    # 자산 핸들 얻기 (scene에 등록된 별칭으로 접근)
    parent = env.scene.get_asset(parent_asset)
    child  = env.scene.get_asset(child_asset)

    if parent is None or child is None:
        raise RuntimeError(f"[attach] asset not found: parent={parent_asset}, child={child_asset}")

    if parent_link not in parent.link_names:
        raise RuntimeError(f"[attach] parent link '{parent_link}' not in {parent.link_names}")

    pose = sim_utils.Pose.create_from_transform(
        p=pose_dict.get("p", [0,0,0]),
        rpy=pose_dict.get("rpy", [0,0,0])
    )

    # 실제 attach (버전에 따라 API 이름이 다를 수 있음: 아래는 일반적인 패턴)
    # Articulation 객체가 제공하는 attach 계열 메서드 사용
    parent.attach_child_articulation(
        child=child,
        parent_link=parent_link,
        pose_in_parent=pose,
        fixed_joint=fixed_joint,
    )
