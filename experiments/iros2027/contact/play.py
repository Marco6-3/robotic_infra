"""Interactive, read-only view of the same physical rollout used in evaluation."""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from threading import Event
import threading

from experiments.iros2027.contact.settings import Settings, load_config

ROOT = Path(__file__).resolve().parents[3]


class PreviewCommand(Exception):
    def __init__(self, action):
        self.action = action


def phase(tick):
    if tick < 700:
        return 'APPROACH'
    if tick < 1400:
        return 'GRASP'
    if tick < 2100:
        return 'LIFT'
    if tick < 2500:
        return 'HOLD'
    if tick < 2800:
        return 'DISTURBANCE'
    return 'RECOVERY'


class Preview:
    """Render a separate model/data pair so GUI edits never affect evaluation."""
    def __init__(self, speed=0.5):
        self.speed = speed
        self.handle = None
        self.paused = Event()
        self.command = None
        self.condition = ''
        self.seed = 0

    def key(self, keycode):
        if keycode == 32:  # Space; keys belong to the viewer, not a terminal.
            if self.paused.is_set():
                self.paused.clear()
            else:
                self.paused.set()
            print('预览：暂停' if self.paused.is_set() else '预览：继续', flush=True)
        elif keycode in (ord('R'), ord('N'), ord('Q')):
            self.command = {ord('R'): 'restart', ord('N'): 'next', ord('Q'): 'quit'}[keycode]
            print(f'预览：{self.command}', flush=True)

    def bind(self, env):
        # Keyboard controls and CLI help do not require the optional GUI runtime.
        import mujoco
        import mujoco.viewer

        if self.handle is None:
            self.model = mujoco.MjModel.from_xml_string(env.xml)
            self.data = mujoco.MjData(self.model)
            self.data.qpos[:] = env.data.qpos
            mujoco.mj_forward(self.model, self.data)
            existing_threads = set(threading.enumerate())
            self.handle = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key,
                                                       show_left_ui=False, show_right_ui=False)
            self.viewer_threads = set(threading.enumerate()) - existing_threads
            with self.handle.lock():
                self.handle.cam.lookat[:] = [.48, 0, .42]
                self.handle.cam.distance = 1.65
                self.handle.cam.azimuth = 135
                self.handle.cam.elevation = -22
                self.handle.opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
                self.handle.opt.sitegroup[:] = 0
        self.started = time.monotonic()
        self.command = None
        self.paused.clear()
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Keep the window/camera across replay and next-condition operations.
        return False

    def check(self):
        if not self.handle.is_running():
            raise PreviewCommand('quit')
        if self.command:
            action = self.command
            self.command = None
            raise PreviewCommand(action)

    def update(self, env, tick, diagnostics):
        import mujoco

        self.check()
        if tick % 16:
            return
        pause_start = time.monotonic()
        while self.paused.is_set():
            self.check()
            self.handle.sync(state_only=True)
            time.sleep(.02)
        self.started += time.monotonic() - pause_start
        deadline = self.started + tick * .001 / self.speed
        while time.monotonic() < deadline:
            self.check()
            time.sleep(min(.01, max(0, deadline - time.monotonic())))
        # Never pass the live physics model/data into the viewer.
        with self.handle.lock():
            self.data.qpos[:] = env.data.qpos
            self.data.qvel[:] = env.data.qvel
            self.data.ctrl[:] = env.data.ctrl
            self.data.time = env.data.time
            mujoco.mj_forward(self.model, self.data)
        normal = env.data.sensordata[:2]
        self.handle.set_texts((None, None,
            'CONTACT PROXY / scripted task\nCondition / seed\nPhase / sim time\nNormal force GT (N)\nGrip residual (mm)\nDropped\nControls',
            f'Preview only; no data saved\n{self.condition} / {self.seed}\n{phase(tick)} / {tick*.001:.2f}s\n'
            f'{normal[0]:.2f} | {normal[1]:.2f}\n{diagnostics["residual_m"]*1000:.1f}\n'
            f'{diagnostics["dropped"]}\nSPACE pause | R restart | N next | Q quit'))
        self.handle.sync(state_only=True)

    def close(self):
        if self.handle is not None:
            self.handle.close()
            # MuJoCo 3.10 launch_passive uses daemon threads. Join the threads
            # created by this window before GLFW's interpreter-exit cleanup.
            for thread in self.viewer_threads:
                thread.join(timeout=5)
                if thread.is_alive():
                    raise RuntimeError('viewer did not finish shutdown')


def main():
    parser = argparse.ArgumentParser(description='观看自动抓取实验；键盘焦点放在 MuJoCo 窗口。')
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/iros2027/contact/config.yaml')
    parser.add_argument('--condition', default='fast')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--speed', type=float, default=.5, help='墙钟播放速度；默认半速，不改变物理步长')
    parser.add_argument('--once', action='store_true', help='运行一个 episode 后关闭窗口')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0 or args.seed < 0:
        parser.error('speed must be positive and seed non-negative')
    config = load_config(args.config)
    settings = Settings.from_mapping(config)
    names = [c['name'] for c in config['conditions']]
    if args.condition not in names:
        parser.error(f'unknown condition; choose from {names}')
    from experiments.iros2027.contact.run import run_episode

    index = names.index(args.condition)
    viewer = Preview(args.speed)
    print('可视化预览：自动循环；空格暂停，R 重来，N 下一条件，Q 退出。按键焦点放在仿真窗口。', flush=True)
    print('不保存实验数据。正式记录请运行 pixi run contact-eval。', flush=True)
    try:
        while True:
            condition = dict(config['conditions'][index])
            viewer.condition = condition.pop('name')
            viewer.seed = args.seed
            try:
                result = run_episode(seed=args.seed, settings=settings, observer_factory=viewer.bind, **condition)
                print(json.dumps({'condition': viewer.condition, 'retained': result['retained'],
                                  'peak_force_n': result['peak_finger_force_n']}), flush=True)
                if args.once:
                    break
                # Keep the final image visible briefly before the next replay.
                for _ in range(50):
                    viewer.check()
                    time.sleep(.02)
            except PreviewCommand as exc:
                if exc.action == 'quit':
                    break
                if exc.action == 'next':
                    index = (index + 1) % len(names)
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
