from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for package_root in ("src/fr3_robot_api", "src/fr3_control", "src/fr3_recorder", "src/fr3_sim"):
    sys.path.insert(0, str(ROOT / package_root))
