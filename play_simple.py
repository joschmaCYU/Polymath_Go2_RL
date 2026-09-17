#!/usr/bin/env python3
"""Simple play / visualization script for Unitree Go2 with keyboard teleoperation.

Allows driving the robot on complex outdoor terrains or flat ground using arrow keys.

Usage:
  python play_simple.py                     # Auto-loads latest checkpoint on Big Map
  python play_simple.py --stage 1           # Play on Stage 1 (Flat ground)
  python play_simple.py --stage 2           # Play on Stage 2 (Stairs & Holes)
  python play_simple.py --stage 3           # Play on Stage 3 (Big Map: stairs, rough, walls, corridors, pits)
  python play_simple.py --checkpoint logs/rsl_rl/go2_velocity/.../model_X.pt
  python play_simple.py --viewer viser      # 3D interactive viewer in web browser
"""

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys
import torch

# Ensure src tasks and robot definitions are registered
import src
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


class KeyboardTeleopController:
  """Contrôleur clavier pour piloter le Go2 avec les flèches directionnelles."""

  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    self.cmd_term = env.unwrapped.command_manager.get_term("twist")
    # Désactiver le rééchantillonnage aléatoire automatique pour garder le contrôle utilisateur
    self.cmd_term.cfg.resampling_time_range = (1e9, 1e9)
    self.vx = 0.0
    self.vy = 0.0
    self.wz = 0.0
    self.v_step = 0.25
    self.w_step = 0.35

  def on_key(self, key: int) -> None:
    from mjlab.viewer.native.keys import (
      KEY_UP,
      KEY_DOWN,
      KEY_LEFT,
      KEY_RIGHT,
      KEY_W,
      KEY_S,
      KEY_A,
      KEY_D,
      KEY_SPACE,
      KEY_X,
    )

    if key in (KEY_UP, KEY_W):
      self.vx = min(self.vx + self.v_step, 2.0)
      print(f"[AVANCER] vx: {self.vx:+.2f} m/s | wz: {self.wz:+.2f} rad/s")
    elif key in (KEY_DOWN, KEY_S):
      self.vx = max(self.vx - self.v_step, -1.0)
      print(f"[RECULER] vx: {self.vx:+.2f} m/s | wz: {self.wz:+.2f} rad/s")
    elif key in (KEY_LEFT, KEY_A):
      self.wz = min(self.wz + self.w_step, 1.8)
      print(f"[GAUCHE ] vx: {self.vx:+.2f} m/s | wz: {self.wz:+.2f} rad/s")
    elif key in (KEY_RIGHT, KEY_D):
      self.wz = max(self.wz - self.w_step, -1.8)
      print(f"[DROITE ] vx: {self.vx:+.2f} m/s | wz: {self.wz:+.2f} rad/s")
    elif key in (KEY_SPACE, KEY_X):
      self.vx = 0.0
      self.vy = 0.0
      self.wz = 0.0
      print(f"[STOP   ] vx: 0.00 m/s | wz: 0.00 rad/s")

    self.apply_command()

  def apply_command(self) -> None:
    self.cmd_term.vel_command_b[:, 0] = self.vx
    self.cmd_term.vel_command_b[:, 1] = self.vy
    self.cmd_term.vel_command_b[:, 2] = self.wz

  def wrap_policy(self, raw_policy):
    def policy_wrapper(obs):
      self.apply_command()
      return raw_policy(obs)

    return policy_wrapper


def parse_args():
  parser = argparse.ArgumentParser(
    description="Visualisation / Play simple pour Unitree Go2 avec contrôle clavier",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "--stage",
    type=str,
    default="3",
    choices=["1", "2", "3", "flat", "medium", "complex"],
    help="Étape de terrain: 1 (plat), 2 (escaliers et trous), 3 (big map complexe)",
  )
  parser.add_argument(
    "--task",
    "-t",
    type=str,
    default=None,
    choices=["complex", "flat", "rough", "stairs_holes"],
    help="Override direct du terrain (ex: complex, flat, stairs_holes)",
  )
  parser.add_argument(
    "--checkpoint",
    "-c",
    type=str,
    default=None,
    help="Chemin vers le checkpoint model_*.pt (par défaut: charge automatiquement le plus récent)",
  )
  parser.add_argument(
    "--num_envs",
    "-n",
    type=int,
    default=1,
    help="Nombre de robots simulés dans le viewer",
  )
  parser.add_argument(
    "--viewer",
    "-v",
    type=str,
    default="auto",
    choices=["auto", "native", "viser"],
    help="Type de visualiseur: auto (détecte écran local ou navigateur), native (MuJoCo GUI), viser (navigateur web 3D)",
  )
  parser.add_argument(
    "--device",
    "-d",
    type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
    help="Device d'exécution PyTorch/MuJoCo",
  )
  return parser.parse_args()


def find_latest_checkpoint(log_root: Path) -> Path | None:
  if not log_root.exists():
    return None
  checkpoints = list(log_root.glob("**/model_*.pt"))
  if not checkpoints:
    return None
  return max(checkpoints, key=lambda p: p.stat().st_mtime)


def main():
  args = parse_args()
  configure_torch_backends()

  stage_to_task = {
    "1": "flat",
    "flat": "flat",
    "2": "stairs_holes",
    "medium": "stairs_holes",
    "stairs_holes": "stairs_holes",
    "3": "complex",
    "complex": "complex",
    "rough": "complex",
  }
  chosen_key = args.task or stage_to_task.get(args.stage, "complex")

  task_map = {
    "flat": "Unitree-Go2-Flat",
    "stairs_holes": "Unitree-Go2-StairsHoles",
    "medium": "Unitree-Go2-StairsHoles",
    "complex": "Unitree-Go2-Complex",
    "rough": "Unitree-Go2-Rough",
  }
  task_id = task_map.get(chosen_key, "Unitree-Go2-Complex")

  stage_desc = {
    "Unitree-Go2-Flat": "Étape 1 : Sol plat",
    "Unitree-Go2-StairsHoles": "Étape 2 : Escaliers et trous modérés",
    "Unitree-Go2-Complex": "Étape 3 : Big Map extérieure (marches, collines, murs, couloirs, trous)",
    "Unitree-Go2-Rough": "Étape 3 : Big Map extérieure (marches, collines, murs, couloirs, trous)",
  }

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = args.num_envs

  # Determine checkpoint
  resume_path = None
  if args.checkpoint:
    resume_path = Path(args.checkpoint).resolve()
    if not resume_path.exists():
      print(f"[ERREUR] Checkpoint specifie introuvable: {resume_path}")
      sys.exit(1)
  else:
    log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    resume_path = find_latest_checkpoint(log_root)

  print("=" * 75)
  print("[PLAY] LANCEMENT DE LA VISUALISATION UNITREE GO2")
  print(f"   * Terrain       : {stage_desc.get(task_id, task_id)}")
  print(f"   * Tache ID      : {task_id}")
  print(f"   * Envs          : {args.num_envs}")
  print(f"   * Device        : {args.device}")
  print(f"   * Viewer        : {args.viewer}")
  if resume_path:
    print(f"   * Checkpoint    : {resume_path}")
  else:
    print("   * Checkpoint    : Aucun (politique neutre par defaut)")
  print("=" * 75)
  print("\n[COMMANDES CLAVIER] (cliquez dans la fenetre graphique pour focus) :")
  print("   * [Fleche HAUT / W]    : Avancer")
  print("   * [Fleche BAS / S]     : Reculer")
  print("   * [Fleche GAUCHE / A]  : Tourner a gauche (rotation yaw)")
  print("   * [Fleche DROITE / D]  : Tourner a droite (rotation yaw)")
  print("   * [Espace / X]         : Stop complet")
  print("   * [Entree]             : Reinitialiser (Reset robot)")
  print("=" * 75 + "\n")

  print("[INIT] Initialisation de l'environnement MuJoCo...")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # Teleop controller setup
  teleop = KeyboardTeleopController(env)

  if resume_path is not None:
    print(f"[LOAD] Chargement du checkpoint : {resume_path.name}...")
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=args.device)
    raw_policy = runner.get_inference_policy(device=args.device)
  else:
    print("[INFO] Mode sans checkpoint entraine. Le robot reste en equilibre de base.")
    action_shape = env.unwrapped.action_space.shape

    class PolicyDefault:
      def __call__(self, obs):
        del obs
        return torch.zeros(action_shape, device=env.unwrapped.device)

    raw_policy = PolicyDefault()

  controlled_policy = teleop.wrap_policy(raw_policy)

  # Viewer selection
  resolved_viewer = args.viewer
  if resolved_viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"

  print(f"[VIEWER] Demarrage du visualiseur [{resolved_viewer.upper()}]...")
  if resolved_viewer == "viser":
    print("[WEB] Ouvrez le lien Viser affiche ci-dessous dans votre navigateur.")

  try:
    if resolved_viewer == "native":
      NativeMujocoViewer(env, controlled_policy, key_callback=teleop.on_key).run()
    elif resolved_viewer == "viser":
      ViserPlayViewer(env, controlled_policy).run()
    else:
      raise ValueError(f"Viewer inconnu: {resolved_viewer}")
  except KeyboardInterrupt:
    print("\n[STOP] Visualisation fermee.")
  finally:
    env.close()
    print("[OK] Session terminee.")


if __name__ == "__main__":
  main()
