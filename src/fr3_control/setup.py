from setuptools import find_packages, setup

package_name = "fr3_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "fr3_robot_api"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    entry_points={"console_scripts": ["fr3_policy_bridge = fr3_control.ros_policy_bridge:main"]},
)
