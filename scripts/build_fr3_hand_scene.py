#!/usr/bin/env python3
"""Build the v1 FR3 + Franka Hand MJCF from pinned source trees.

Menagerie's FR3 and Franka Hand are intentionally separate models. This
generator keeps both upstream inputs untouched, copies their assets into one
model directory, attaches the hand at the FR3 attachment site, and adds direct
the single tendon position actuator required for the continuous mimic gripper.
"""

from __future__ import annotations

import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


PREFIX = "fr3_"
HAND_CLASS_PREFIX = "fr3_hand_"
HAND_MATERIAL_PREFIX = "fr3_hand_"


def _copy_assets(sources: list[tuple[Path, set[str]]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source, needed in sources:
        for item in source.iterdir():
            if not item.is_file():
                continue
            # The Menagerie Panda directory also contains the complete Panda
            # arm asset set. Copy only files actually referenced by hand.xml;
            # otherwise names such as link0_2.obj collide with FR3 assets.
            if item.name not in needed:
                continue
            target = destination / item.name
            if target.exists():
                if target.read_bytes() != item.read_bytes():
                    raise RuntimeError(f"asset name collision with different contents: {target}")
            else:
                shutil.copy2(item, target)


def _prefix_hand_references(root: ET.Element) -> None:
    # The FR3 and Panda hand models both define generic MuJoCo default class
    # names (`visual`, `collision`, ...).  MuJoCo stores defaults in one model
    # namespace, so simply appending the hand defaults to the arm would make
    # the composite XML fail with "repeated default class".  Rename the hand
    # classes before the two trees are merged and update every reference to
    # them (including `childclass` inherited by bodies).
    hand_classes = {
        element.get("class")
        for element in root.findall(".//default")
        if element.get("class")
    }
    class_map = {name: HAND_CLASS_PREFIX + name for name in hand_classes}
    for element in root.iter():
        for key in ("class", "childclass"):
            value = element.get(key)
            if value in class_map:
                element.set(key, class_map[value])

    # Material names also share one global namespace after the asset sections
    # are merged.  Keep the hand's appearance independent from any arm
    # material with the same upstream name (notably `black` and `white`).
    material_map = {
        element.get("name"): HAND_MATERIAL_PREFIX + element.get("name")
        for element in root.findall("./asset/material")
        if element.get("name")
    }
    for element in root.iter():
        if element.tag == "material" and element.get("name") in material_map:
            element.set("name", material_map[element.get("name")])
        material = element.get("material")
        if material in material_map:
            element.set("material", material_map[material])

    hand_joint_names = {"finger_joint1", "finger_joint2"}
    body_names = {"hand", "left_finger", "right_finger"}
    for element in root.iter():
        if element.tag == "body" and element.get("name") in body_names:
            element.set("name", PREFIX + element.get("name"))
        if element.tag == "joint" and element.get("name") in hand_joint_names:
            element.set("name", PREFIX + element.get("name"))
            # Keep the physical continuous finger range explicit in the
            # composite XML.  The upstream hand inherits it from the `finger`
            # default class, while ROS/validation tooling reads joint ranges.
            element.set("range", "0 0.04")
        if element.tag == "fixed" and element.get("name") == "split":
            element.set("name", "fr3_split")
        for key in ("joint", "joint1", "joint2", "body1", "body2", "tendon"):
            value = element.get(key)
            if value in hand_joint_names or value in body_names or value == "split":
                element.set(key, PREFIX + value)


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    raise RuntimeError(f"could not find body {name!r}")


def _read_fr3_joint_limits(path: Path) -> dict[str, tuple[float, float]]:
    """Read the small, stable `robots/fr3/joint_limits.yaml` subset.

    Avoid making the model generator depend on PyYAML: the pinned upstream
    file is deliberately a flat mapping of joint1..joint7 to `limit` values.
    """
    current: str | None = None
    limits: dict[str, tuple[float, float]] = {}
    for line in path.read_text().splitlines():
        joint = re.match(r"^joint([1-7]):\s*$", line)
        if joint:
            current = f"fr3_joint{joint.group(1)}"
            continue
        if current is None:
            continue
        lower = re.match(r"^\s*lower:\s*([-+0-9.eE]+)", line)
        upper = re.match(r"^\s*upper:\s*([-+0-9.eE]+)", line)
        if lower:
            old = limits.get(current, (0.0, 0.0))
            limits[current] = (float(lower.group(1)), old[1])
        elif upper:
            old = limits.get(current, (0.0, 0.0))
            limits[current] = (old[0], float(upper.group(1)))
    expected = {f"fr3_joint{i}" for i in range(1, 8)}
    if set(limits) != expected:
        raise RuntimeError(f"invalid FR3 joint limits file {path}: expected {sorted(expected)}")
    return limits


def _apply_fr3_joint_limits(root: ET.Element, limits: dict[str, tuple[float, float]]) -> None:
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if name in limits:
            lower, upper = limits[name]
            joint.set("range", f"{lower:g} {upper:g}")


def build(
    menagerie_root: Path,
    output_root: Path,
    fr3_joint_limits: Path | None = None,
) -> Path:
    arm_root = menagerie_root / "franka_fr3"
    hand_root = menagerie_root / "franka_emika_panda"
    arm_xml = ET.parse(arm_root / "fr3.xml")
    hand_xml = ET.parse(hand_root / "hand.xml")
    arm = arm_xml.getroot()
    hand = hand_xml.getroot()
    _prefix_hand_references(hand)

    output_root.mkdir(parents=True, exist_ok=True)
    arm_meshes = {
        Path(mesh.get("file")).name
        for mesh in arm.findall(".//mesh")
        if mesh.get("file")
    }
    hand_meshes = {
        Path(mesh.get("file")).name
        for mesh in hand.findall(".//mesh")
        if mesh.get("file")
    }
    _copy_assets(
        [(arm_root / "assets", arm_meshes), (hand_root / "assets", hand_meshes)],
        output_root / "assets",
    )

    # The arm compiler already provides the common mesh directory. Keep the
    # arm defaults and append the hand's namespaced default classes.
    arm_compiler = arm.find("compiler")
    if arm_compiler is not None:
        arm_compiler.set("meshdir", "assets")
    option = arm.find("option")
    if option is None:
        option = ET.SubElement(arm, "option")
    option.set("timestep", "0.001")
    option.set("integrator", option.get("integrator", "implicitfast"))
    if fr3_joint_limits is not None:
        _apply_fr3_joint_limits(arm, _read_fr3_joint_limits(fr3_joint_limits))
    for tag in ("default", "asset"):
        destination = arm.find(tag)
        source = hand.find(tag)
        if destination is not None and source is not None:
            destination.extend(list(source))

    arm_worldbody = arm.find("worldbody")
    hand_worldbody = hand.find("worldbody")
    if arm_worldbody is None or hand_worldbody is None:
        raise RuntimeError("both source models must define worldbody")
    hand_body = next(iter(hand_worldbody.findall("body")), None)
    if hand_body is None:
        raise RuntimeError("hand model has no root body")
    hand_body.set("pos", "0 0 0.107")
    # This is the flange-to-hand conversion-frame rotation used by the pinned
    # Menagerie Panda attachment body. The resulting TCP transform is checked
    # against the official franka_description FK in the validation step.
    # MuJoCo uses (w, x, y, z) quaternions.  This is -pi/4 about Z, matching
    # the official franka_description `fr3_hand_joint` rpy and Menagerie's
    # complete Panda model (the Panda no-hand attachment uses the inverse
    # conversion frame and must not be copied here).
    hand_body.set("quat", "0.9238795 0 0 -0.3826834")
    # The official hand TCP is 103.4 mm from the hand frame.
    hand_body.append(ET.Element("site", {"name": "fr3_hand_tcp", "pos": "0 0 0.1034", "size": "0.002"}))
    # A fixed wrist RGB camera. Rendering parameters are supplied by the
    # runtime camera adapter; this pose is part of the reproducible model.
    hand_body.append(
        ET.Element(
            "camera",
            {
                "name": "wrist",
                "pos": "0.06 0 0.035",
                "xyaxes": "0 1 0 0 0 1",
                "fovy": "42.5",
                "resolution": "640 480",
            },
        )
    )
    fr3_link7 = _find_body(arm, "fr3_link7")
    fr3_link7.append(hand_body)

    # Preserve the hand's coupling constraint and expose one tendon actuator.
    # mujoco_ros2_control maps mimic grippers through one actuated tendon;
    # finger_joint2 remains simulation-only state constrained by equality.
    for tag in ("contact", "tendon", "equality"):
        source = hand.find(tag)
        if source is not None:
            arm_destination = arm.find(tag)
            if arm_destination is None:
                arm_destination = ET.SubElement(arm, tag)
            arm_destination.extend(list(source))
    actuators = arm.find("actuator")
    if actuators is None:
        actuators = ET.SubElement(arm, "actuator")
    ET.SubElement(
        actuators,
        "position",
        {
            "name": "fr3_finger_joint1",
            "tendon": "fr3_split",
            "kp": "1000",
            "kv": "20",
            "ctrlrange": "0 0.04",
        },
    )

    # Adding two finger DOFs and one tendon actuator changes nq and nu. Extend
    # the upstream arm-only keyframe so the composite model can compile.
    for key in arm.findall(".//keyframe/key"):
        qpos = key.get("qpos")
        ctrl = key.get("ctrl")
        if qpos:
            key.set("qpos", f"{qpos} 0.04 0.04")
        if ctrl:
            key.set("ctrl", f"{ctrl} 0.04")

    ET.indent(arm, space="  ")
    composite = output_root / "fr3_hand.xml"
    ET.ElementTree(arm).write(composite, encoding="utf-8", xml_declaration=True)

    scene = ET.Element("mujoco", {"model": "fr3 hand scene"})
    ET.SubElement(scene, "include", {"file": "fr3_hand.xml"})
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})
    asset = ET.SubElement(scene, "asset")
    ET.SubElement(asset, "texture", {"type": "2d", "name": "groundplane", "builtin": "checker", "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300"})
    ET.SubElement(asset, "material", {"name": "groundplane", "texture": "groundplane", "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2"})
    worldbody = ET.SubElement(scene, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(worldbody, "geom", {"name": "floor", "size": "0 0 0.05", "type": "plane", "material": "groundplane"})
    ET.SubElement(worldbody, "camera", {"name": "external", "pos": "1.2 -1.2 0.9", "xyaxes": "0.707 0.707 0 -0.25 0.25 0.935", "fovy": "42.5", "resolution": "640 480"})
    ET.indent(scene, space="  ")
    scene_path = output_root / "scene.xml"
    ET.ElementTree(scene).write(scene_path, encoding="utf-8", xml_declaration=True)
    return scene_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--menagerie-root", type=Path, default=Path("third_party/mujoco_menagerie"))
    parser.add_argument("--output", type=Path, default=Path("src/fr3_description/models"))
    parser.add_argument(
        "--fr3-joint-limits",
        type=Path,
        default=Path("third_party/franka_description/robots/fr3/joint_limits.yaml"),
        help="pinned official franka_description limits used to keep MJCF and URDF compatible",
    )
    args = parser.parse_args()
    limits = args.fr3_joint_limits if args.fr3_joint_limits.exists() else None
    path = build(args.menagerie_root, args.output, limits)
    print(f"generated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
