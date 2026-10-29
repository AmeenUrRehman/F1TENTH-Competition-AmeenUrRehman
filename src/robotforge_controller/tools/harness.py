#!/usr/bin/env python3
"""Headless lap tester: runs the controller logic in the official simulator.

Uses the same f1tenth_gym backend, map loader, LiDAR settings, 100 Hz physics
and lap/collision bookkeeping as the competition bridge, but without ROS or
RViz, and faster than real time. The controller is the exact ``Racer`` class
the ROS node uses, called every ``--ctrl-period`` seconds on a scan that is
``--latency`` physics steps old.

Run with the competition venv's Python, e.g.:
  starter_ws/.venv/bin/python tools/harness.py --map levine_blocked \\
      --mode pursuit --raceline racelines/levine_blocked_raceline.csv --trials 5
"""

import argparse
import collections
import importlib.util
import math
import pathlib
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # the robotforge_controller python package

COMPETITION = pathlib.Path.home() / "CompetionF1" / \
    "CIE-KFUPM-F1tenth-Autonomous-Racing-Competition"
SIM_ROS = COMPETITION / "starter_ws" / "src" / "f1tenth_gym_ros"


def load_bridge_module():
    """Import gym_bridge.py for its map helpers, stubbing ROS-only imports."""
    for name in ("ackermann_msgs", "ackermann_msgs.msg"):
        if importlib.util.find_spec(name.split(".")[0]) is None:
            mod = types.ModuleType(name)
            mod.AckermannDriveStamped = object
            sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location(
        "gym_bridge", SIM_ROS / "f1tenth_gym_ros" / "gym_bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_env(bridge, map_yaml):
    from f1tenth_gym.envs import F110Env
    from f1tenth_gym.envs.action import LongitudinalActionType, SteerActionType
    from f1tenth_gym.envs.dynamic_models import (DynamicModel,
                                                 get_f1tenth_vehicle_parameters)
    from f1tenth_gym.envs.env_config import (ControlConfig, EnvConfig,
                                             LoopCounterMode, ObservationConfig,
                                             ResetConfig, SimulationConfig)
    from f1tenth_gym.envs.integrators import IntegratorType
    from f1tenth_gym.envs.lidar import LiDARConfig
    from f1tenth_gym.envs.observation import ObservationType
    from f1tenth_gym.envs.reset import ResetStrategy
    from f1tenth_gym.envs.track import Track

    try:
        track = Track.from_track_path(map_yaml, track_scale=1.0)
        bridge._attach_yaml_reference_lines(track, map_yaml, 1.0)
    except (ValueError, FileNotFoundError):
        track, _ = bridge._load_track_from_yaml(map_yaml, 1.0)

    lidar = LiDARConfig(enabled=True, num_beams=819, range_min=0.05,
                        range_max=25.0, angle_min=np.deg2rad(-135.0),
                        angle_max=np.deg2rad(135.0), noise_std=0.01,
                        base_link_to_lidar_tf=(0.275, 0.0, 0.0))
    cfg = EnvConfig(
        map_name=track, map_scale=1.0,
        params=get_f1tenth_vehicle_parameters(), num_agents=1,
        control_config=ControlConfig(
            longitudinal_mode=LongitudinalActionType.SPEED,
            steering_mode=SteerActionType.STEERING_ANGLE),
        simulation_config=SimulationConfig(
            timestep=0.01, integrator_timestep=0.01,
            integrator=IntegratorType.RK4, dynamics_model=DynamicModel.ST,
            loop_counter=LoopCounterMode.FRENET_BASED,
            compute_frenet_frame=True, max_laps=None,
            count_partial_first_lap=False),
        observation_config=ObservationConfig(type=ObservationType.DEFAULT),
        reset_config=ResetConfig(strategy=ResetStrategy.MAP_RANDOM_STATIC),
        lidar_config=lidar, render_enabled=False)
    return F110Env(config=cfg, render_mode=None), lidar


def run_trial(env, lidar, racer, start, args):
    lr = 0.17145
    sx, sy, st = start
    cog = [sx + lr * math.cos(st), sy + lr * math.sin(st), st]
    env.reset(options={"poses": np.array([cog])})
    racer.reset()
    sim = env.unwrapped
    ctrl_every = max(1, round(args.ctrl_period / 0.01))
    history = collections.deque(maxlen=args.latency + 1)
    action = np.array([[0.0, 0.0]])
    laps, last_count = [], 0
    trace = []
    t = 0.0
    while t < args.timeout:
        state = sim.sim.state
        pose = state.poses[0]
        speed = float(state.standard_state[0, 3])
        history.append((np.array(state.scans[0]), pose.copy(), speed))
        if bool(state.collisions[0]):
            return laps, (float(pose[0]), float(pose[1])), t, trace
        count = int(sim.lap_counts[0])
        if count > last_count:
            laps.append(float(sim.lap_times[0]))
            last_count = count
            if len(laps) >= args.laps:
                return laps, None, t, trace
        step = int(round(t / 0.01))
        if step % ctrl_every == 0:
            scan, p, v = history[0]
            base = (p[0] - lr * math.cos(p[2]), p[1] - lr * math.sin(p[2]), p[2])
            steer, cmd = racer.compute(scan, lidar.angle_min,
                                       lidar.angle_increment, v, base)
            action = np.array([[steer, cmd]])
            if args.trace:
                trace.append((t, base[0], base[1], v, steer, cmd))
        env.step(action)
        t += 0.01
    return laps, "timeout", t, trace


def main():
    from robotforge_controller.pure_pursuit import load_raceline
    from robotforge_controller.racer import Racer

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="levine",
                    help="map name in the simulator's maps/ folder, or a yaml path")
    ap.add_argument("--mode", choices=["gap", "pursuit"], default="gap")
    ap.add_argument("--raceline")
    ap.add_argument("--max-speed", type=float, default=6.0)
    ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--start", type=float, nargs=3, default=[-12.0, 0.0, 0.0],
                    metavar=("X", "Y", "YAW"), help="base_link start pose")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="random start offset (m, and x0.1 rad) per trial")
    ap.add_argument("--ctrl-period", type=float, default=0.02)
    ap.add_argument("--latency", type=int, default=2,
                    help="physics steps between scan and command")
    ap.add_argument("--timeout", type=float, default=300.0, help="sim seconds")
    ap.add_argument("--trace", help="write the last trial's trajectory CSV here")
    args = ap.parse_args()

    map_yaml = pathlib.Path(args.map)
    if map_yaml.suffix not in (".yaml", ".yml"):
        map_yaml = SIM_ROS / "maps" / f"{args.map}.yaml"
    bridge = load_bridge_module()
    env, lidar = make_env(bridge, map_yaml)
    raceline = load_raceline(args.raceline) if args.raceline else None

    rng = np.random.default_rng(0)
    clean, totals = 0, []
    for i in range(args.trials):
        racer = Racer(args.mode, args.max_speed, raceline, args.speed_scale)
        start = np.array(args.start, dtype=float)
        if args.jitter:
            start += rng.uniform(-1, 1, 3) * np.array([args.jitter, args.jitter * 0.3,
                                                       args.jitter * 0.1])
        laps, crash, t, trace = run_trial(env, lidar, racer, start, args)
        if crash is None:
            clean += 1
            totals.append(sum(laps))
            status = f"CLEAN total {sum(laps):6.2f} s"
        elif crash == "timeout":
            status = f"TIMEOUT after {t:.0f} s"
        else:
            status = f"CRASH at ({crash[0]:.2f}, {crash[1]:.2f}) t={t:.1f} s"
        print(f"trial {i + 1}: laps {[round(x, 2) for x in laps]}  {status}", flush=True)
        if args.trace and trace:
            np.savetxt(args.trace, np.array(trace), delimiter=",", fmt="%.4f",
                       header="t,x,y,v,steer,cmd")
    best = f", best total {min(totals):.2f} s" if totals else ""
    print(f"SUMMARY {map_yaml.stem} {args.mode}: {clean}/{args.trials} clean{best}")


if __name__ == "__main__":
    main()
