"""Validated experiment configuration shared by visual and headless entrypoints."""
from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
import re
import yaml


@dataclass(frozen=True)
class TaskSettings:
    mass_range_kg: tuple = (.055, .075)
    friction_range: tuple = (.65, .85)
    disturbance_range_n: tuple = (5.5, 6.5)


@dataclass(frozen=True)
class SensorSettings:
    hz: int = 200
    noise_std_n: float = .01
    max_age_ms: float = 200


@dataclass(frozen=True)
class ControllerSettings:
    fast_hz: int = 100
    semantic_hz: int = 5
    semantic_max_age_ms: float = 250
    max_closure_m: float = .016
    step_m: float = .002
    force_limit_n: float = 12.
    shear_ratio_threshold: float = .45


@dataclass(frozen=True)
class Settings:
    task: TaskSettings = field(default_factory=TaskSettings)
    sensor: SensorSettings = field(default_factory=SensorSettings)
    controller: ControllerSettings = field(default_factory=ControllerSettings)

    @classmethod
    def from_mapping(cls, raw):
        result = cls(**{name: typ(**raw.get(name, {})) for name, typ in
                        [('task', TaskSettings), ('sensor', SensorSettings), ('controller', ControllerSettings)]})
        for pair in asdict(result.task).values():
            if len(pair) != 2 or not all(math.isfinite(x) for x in pair) or not 0 < pair[0] <= pair[1]:
                raise ValueError('physics ranges must be finite, positive [min, max]')
        for hz in (result.sensor.hz, result.controller.fast_hz, result.controller.semantic_hz):
            if isinstance(hz, bool) or not isinstance(hz, int) or hz <= 0 or 1000 % hz:
                raise ValueError('rates must be positive integer divisors of the 1000 Hz physics rate')
        for group in (result.sensor, result.controller):
            for name, value in asdict(group).items():
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f'{name} must be finite and non-negative')
        c = result.controller
        if min(c.max_closure_m, c.step_m, c.force_limit_n, c.shear_ratio_threshold) <= 0 or c.max_closure_m > .08:
            raise ValueError('invalid residual bounds')
        return result


def load_config(path: Path):
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) - {'seeds', 'conditions', 'task', 'sensor', 'controller'}:
        raise ValueError('unknown configuration keys')
    Settings.from_mapping(raw)
    seeds = raw.get('seeds', [])
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be unique non-negative integers')
    conditions = raw.get('conditions', [])
    names = []
    for condition in conditions:
        if set(condition) - {'name', 'mode', 'sensor_delay_ms', 'semantic_delay_ms', 'disturbance_scale'}:
            raise ValueError('unknown condition keys')
        name = condition.get('name', '')
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
            raise ValueError('condition names must contain only letters, digits, underscore or hyphen')
        if condition.get('mode') not in ('nominal', 'fast'):
            raise ValueError('mode must be nominal or fast')
        for key in ('sensor_delay_ms', 'semantic_delay_ms', 'disturbance_scale'):
            value = condition.get(key, 0)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and non-negative')
        names.append(name)
    if not names or len(names) != len(set(names)):
        raise ValueError('conditions must be non-empty and uniquely named')
    return raw
