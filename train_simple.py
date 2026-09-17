#!/usr/bin/env python3
"""Simple training script for Unitree Go2 with Reinforcement Learning.

Default training is set on Complex Outdoor Terrains (marches, terrains accidentés,
murs, longs couloirs, trous).

Usage:
  python train_simple.py
  python train_simple.py --task complex --num_envs 1024
  python train_simple.py --task flat
  python train_simple.py --resume
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path
import sys
import torch

# Ensure src tasks and robot definitions are registered
import src
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


def parse_args():
  parser = argparse.ArgumentParser(
    description="Entraînement RL simple pour Unitree Go2",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "--stage",
    type=str,
    default="3",
    choices=["1", "2", "3", "flat", "medium", "complex"],
    help="Étape d'apprentissage progressif: 1 (plat), 2 (escaliers et trous), 3 (big map complexe)",
  )
  parser.add_argument(
    "--task",
    "-t",
    type=str,
    default=None,
    choices=["complex", "flat", "rough", "stairs_holes"],
    help="Type de terrain pour l'entraînement (override de --stage)",
  )
  parser.add_argument(
    "--num_envs",
    "-n",
    type=int,
    default=4096,
    help="Nombre d'environnements parallèles simulés (4096 calibré pour RTX 4070 8 Go, 1024-2048 pour 6 Go)",
  )
  parser.add_argument(
    "--max_iterations",
    "-i",
    type=int,
    default=3000,
    help="Nombre total d'itérations d'apprentissage",
  )
  parser.add_argument(
    "--save_interval",
    "-s",
    type=int,
    default=50,
    help="Intervalle de sauvegarde des checkpoints (en itérations)",
  )
  parser.add_argument(
    "--resume",
    action="store_true",
    help="Reprendre l'entraînement depuis le dernier checkpoint disponible",
  )
  parser.add_argument(
    "--checkpoint",
    "-c",
    type=str,
    default=None,
    help="Chemin spécifique vers un checkpoint .pt pour reprendre",
  )
  parser.add_argument(
    "--device",
    "-d",
    type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
    help="Device d'exécution PyTorch/MuJoCo",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Graine aléatoire",
  )
  parser.add_argument(
    "--logger",
    type=str,
    default="tensorboard",
    choices=["tensorboard", "wandb", "neptune"],
    help="Type de logger (tensorboard par défaut, pas besoin de compte en ligne)",
  )
  return parser.parse_args()


def get_latest_checkpoint(log_root: Path) -> Path | None:
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
    "Unitree-Go2-Flat": "Étape 1 : Sol plat (fondations locomotion)",
    "Unitree-Go2-StairsHoles": "Étape 2 : Escaliers et trous modérés",
    "Unitree-Go2-Complex": "Étape 3 : Big Map extérieure (marches, collines, murs, couloirs, trous)",
    "Unitree-Go2-Rough": "Étape 3 : Big Map extérieure (marches, collines, murs, couloirs, trous)",
  }

  print("=" * 75)
  print(f"[TRAIN] LANCEMENT DE L'ENTRAINEMENT UNITREE GO2")
  print(f"   * Niveau    : {stage_desc.get(task_id, task_id)}")
  print(f"   * Tache ID  : {task_id}")
  print(f"   * Envs      : {args.num_envs}")
  print(f"   * Device    : {args.device}")
  print(f"   * Iterations: {args.max_iterations}")
  print(f"   * Seed      : {args.seed}")
  print("=" * 75)

  if "cpu" in str(args.device) or not torch.cuda.is_available():
    print("\n" + "!" * 80)
    print("[ATTENTION CRITIQUE] L'ENTRAINEMENT S'EXECUTE SUR CPU ET NON SUR GPU CUDA !")
    print("   * Sur CPU, la simulation de 4096 robots prend ~132s par pas de collecte (19h l'etape 1).")
    print("   * Sur GPU CUDA, cette meme etape ne prend que ~2.0s par iteration (4 min l'etape 1).")
    print("   * Diagnostic si vous utilisez Docker :")
    print("       1. Sur la machine hote, verifiez que le pilote repond : nvidia-smi")
    print("       2. Verifiez que nvidia-container-toolkit est installe et configure pour Docker :")
    print("          sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker")
    print("       3. Testez dans Docker : docker compose run --rm go2-rl nvidia-smi")
    print("!" * 80 + "\n")

  # Load configurations
  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)

  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  agent_cfg.seed = args.seed
  agent_cfg.save_interval = args.save_interval
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.logger = args.logger

  # Setup logging directory
  log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_dir = log_root / f"{timestamp}_{args.task}"
  log_dir.mkdir(parents=True, exist_ok=True)
  (log_dir / "params").mkdir(parents=True, exist_ok=True)

  # Checkpoint resolution
  resume_path = None
  if args.checkpoint:
    resume_path = Path(args.checkpoint).resolve()
    if not resume_path.exists():
      print(f"[ERREUR] Checkpoint introuvable: {resume_path}")
      sys.exit(1)
  elif args.resume:
    resume_path = get_latest_checkpoint(log_root)
    if resume_path:
      print(f"[REPRISE] Reprise depuis le dernier checkpoint: {resume_path}")
    else:
      print("[INFO] Aucun checkpoint existant trouve, demarrage d'un nouvel entrainement.")

  print("\n[INIT] Creation de l'environnement de simulation...")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # Save parameter configs
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner_cls = load_runner_cls(task_id)
  runner = runner_cls(env, asdict(agent_cfg), str(log_dir), args.device)

  if resume_path is not None:
    print(f"[LOAD] Chargement des poids depuis {resume_path}")
    runner.load(str(resume_path))

  print(f"[LOGS] Logs et checkpoints enregistres dans: {log_dir}\n")
  print("[START] Debut de la boucle d'apprentissage RL...")
  try:
    runner.learn(
      num_learning_iterations=args.max_iterations,
      init_at_random_ep_len=True,
    )
  except KeyboardInterrupt:
    print("\n[STOP] Entrainement interrompu par l'utilisateur.")
  finally:
    env.close()
    print("[OK] Environnement ferme proprement.")


if __name__ == "__main__":
  main()
