#!/usr/bin/env python3
"""Script d'entrainement automatise en 3 etapes (Curriculum Learning) pour Unitree Go2.

Ce script enchaine automatiquement les 3 etapes d'apprentissage :
  Etape 1 : Sol plat (fondations de la marche et equilibre)
  Etape 2 : Escaliers et trous (franchissement d'obstacles moderes)
  Etape 3 : Big Map complexe (marches, collines, murs, couloirs, fosses profondes)

Chaque etape beneficie de la supervision auto-recovery :
  - Relance automatique en cas de plantage (jusqu'a 10 relances).
  - Protection contre les boucles infinies si un crash survient en moins de 30 secondes.
  - Transmission automatique du dernier checkpoint a l'etape suivante.

Usage simple (calibre pour RTX 4070 8 Go - duree totale ~25 a 30 min) :
  python train_all.py

Usage avance :
  python train_all.py --num_envs 2048  # Si execution sur carte 6 Go
  python train_all.py --from_stage 2
  python train_all.py --resume
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def find_latest_checkpoint(log_dir: Path) -> Path | None:
  """Recherche le checkpoint model_*.pt le plus recent."""
  if not log_dir.exists():
    return None
  checkpoints = list(log_dir.glob("**/model_*.pt"))
  if not checkpoints:
    return None
  return max(checkpoints, key=lambda p: p.stat().st_mtime)


def run_stage_with_recovery(
  stage_num: int,
  task_name: str,
  stage_description: str,
  iterations: int,
  num_envs: int,
  initial_checkpoint: str | None,
  max_restarts: int = 10,
  cooldown: float = 3.0,
  log_root: Path = Path("logs") / "rsl_rl" / "go2_velocity",
) -> str | None:
  """Execute une etape avec reprise automatique en cas de crash.

  Retourne le chemin du dernier checkpoint genere par l'etape.
  """
  current_checkpoint = initial_checkpoint
  restart_count = 0
  consecutive_rapid_failures = 0
  interrupted = False

  print("\n" + "=" * 80)
  print(f"[CURRICULUM] ETAPE {stage_num}/3 : {stage_description.upper()}")
  print(f"   * Tache          : {task_name}")
  print(f"   * Iterations     : {iterations}")
  print(f"   * Environnements : {num_envs}")
  if current_checkpoint:
    print(f"   * Poids initiaux : {current_checkpoint}")
  else:
    print(f"   * Poids initiaux : Aucun (depart de zero)")
  print("=" * 80)

  while restart_count < max_restarts and not interrupted:
    cmd = [
      sys.executable,
      "train_simple.py",
      "--task",
      task_name,
      "--num_envs",
      str(num_envs),
      "--max_iterations",
      str(iterations),
      "--save_interval",
      "50",
    ]

    if current_checkpoint:
      cmd.extend(["--checkpoint", str(current_checkpoint)])
    elif restart_count > 0:
      cmd.append("--resume")

    now_str = datetime.now().strftime("%H:%M:%S")
    if restart_count > 0:
      print(f"\n[{now_str}] [RECOVERY ETAPE {stage_num} #{restart_count}/{max_restarts}] Relance de l'entrainement...")
      if current_checkpoint:
        print(f"[{now_str}] [REPRISE] Checkpoint : {current_checkpoint}")
    else:
      print(f"[{now_str}] [LANCEMENT] Demarrage de l'etape {stage_num}...")

    start_time = time.time()

    try:
      proc = subprocess.Popen(cmd)

      def handle_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n\n[ARRET] Interruption manuelle demandee a l'etape {stage_num}. Fermeture du processus...")
        proc.terminate()
        try:
          proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
          proc.kill()
        sys.exit(0)

      signal.signal(signal.SIGINT, handle_sigint)
      exit_code = proc.wait()
      duration = time.time() - start_time

      if exit_code == 0:
        now_str = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now_str}] [OK] Etape {stage_num} terminee avec succes !")
        latest = find_latest_checkpoint(log_root)
        if latest:
          print(f"[{now_str}] [CHECKPOINT] Dernier modele sauvegarde : {latest}")
          return str(latest)
        return current_checkpoint

      else:
        if interrupted:
          return None

        restart_count += 1
        now_str = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now_str}] [ATTENTION] Etape {stage_num} interrompue (Code: {exit_code}, duree: {duration:.1f}s).")

        # Securite anti-boucle : crash rapide (< 30 secondes)
        if duration < 30.0:
          consecutive_rapid_failures += 1
          print(f"[{now_str}] [ATTENTION] Detection d'un crash precoce ({consecutive_rapid_failures}/2 consecutifs).")
          if consecutive_rapid_failures >= 2:
            print("\n" + "!" * 80)
            print(f"[ERREUR FATALE] Crash repete au demarrage de l'etape {stage_num} (duree < 30s).")
            print("   Arret du curriculum pour eviter une boucle infinie.")
            print("   Consultez les messages d'erreur au-dessus.")
            print("!" * 80)
            sys.exit(1)
        else:
          consecutive_rapid_failures = 0

        if restart_count >= max_restarts:
          print(f"\n[ERREUR] Plafond de {max_restarts} relances atteint pour l'etape {stage_num}. Arret.")
          sys.exit(1)

        latest = find_latest_checkpoint(log_root)
        if latest:
          current_checkpoint = str(latest)
          print(f"[{now_str}] [CHECKPOINT] Reprise au checkpoint : {current_checkpoint}")

        print(f"[PAUSE] Pause de {cooldown}s avant relance...")
        time.sleep(cooldown)

    except KeyboardInterrupt:
      interrupted = True
      print(f"\n[FIN] Arret manuel de l'etape {stage_num}.")
      return None

  return None


def parse_args():
  parser = argparse.ArgumentParser(
    description="Entrainement complet Unitree Go2 en 3 etapes automatiques",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "--num_envs",
    "-n",
    type=int,
    default=4096,
    help="Nombre d'environnements paralleles (4096 calibre pour RTX 4070 8 Go, 1024-2048 pour 6 Go)",
  )
  parser.add_argument(
    "--from_stage",
    "-s",
    type=int,
    default=1,
    choices=[1, 2, 3],
    help="Etape de depart du curriculum (1: Plat, 2: Escaliers/Trous, 3: Big Map)",
  )
  parser.add_argument(
    "--resume",
    "-r",
    action="store_true",
    help="Reprendre l'entrainement depuis le dernier checkpoint disponible (meme pour l'etape 1)",
  )
  parser.add_argument(
    "--checkpoint",
    "-c",
    type=str,
    default=None,
    help="Checkpoint initial optionnel pour debuter",
  )
  parser.add_argument(
    "--iters_stage1",
    type=int,
    default=500,
    help="Nombre d'iterations pour l'etape 1 (Sol plat, ~49M steps avec 4096 envs)",
  )
  parser.add_argument(
    "--iters_stage2",
    type=int,
    default=750,
    help="Nombre d'iterations pour l'etape 2 (Escaliers et trous, ~73M steps avec 4096 envs)",
  )
  parser.add_argument(
    "--iters_stage3",
    type=int,
    default=1200,
    help="Nombre d'iterations pour l'etape 3 (Big Map complexe, ~118M steps avec 4096 envs)",
  )
  parser.add_argument(
    "--max_restarts",
    type=int,
    default=10,
    help="Nombre max de relances automatiques par etape en cas de crash",
  )
  parser.add_argument(
    "--cooldown",
    type=float,
    default=3.0,
    help="Duree de temporisation (en secondes) avant relance apres crash",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  log_root = Path("logs") / "rsl_rl" / "go2_velocity"

  stages = [
    {
      "stage_num": 1,
      "task_name": "flat",
      "description": "Sol plat (fondations locomotion)",
      "iterations": args.iters_stage1,
    },
    {
      "stage_num": 2,
      "task_name": "stairs_holes",
      "description": "Escaliers et trous moderes",
      "iterations": args.iters_stage2,
    },
    {
      "stage_num": 3,
      "task_name": "complex",
      "description": "Big Map exterieure (marches, collines, murs, couloirs, trous)",
      "iterations": args.iters_stage3,
    },
  ]

  print("=" * 80)
  print("[PIPELINE] ENTRAINEMENT AUTOMATIQUE COMPLET EN 3 ETAPES - UNITREE GO2")
  print(f"   * Depart       : Etape {args.from_stage}/3")
  print(f"   * Envs         : {args.num_envs}")
  print(f"   * Iterations S1: {args.iters_stage1} | S2: {args.iters_stage2} | S3: {args.iters_stage3}")
  print(f"   * Total prevu  : {args.iters_stage1 + args.iters_stage2 + args.iters_stage3} iterations")
  print("=" * 80)
  print("[INFO] Appuyez sur Ctrl+C a tout moment pour interrompre l'execution.\n")

  current_checkpoint = args.checkpoint
  if not current_checkpoint and (args.resume or args.from_stage > 1):
    latest = find_latest_checkpoint(log_root)
    if latest:
      current_checkpoint = str(latest)
      print(f"[INFO] Reprise automatique depuis le dernier checkpoint disponible : {current_checkpoint}")
    elif args.resume:
      print("[INFO] Aucun checkpoint existant trouve pour --resume, demarrage a zero.")

  for stage_info in stages:
    if stage_info["stage_num"] < args.from_stage:
      continue

    checkpoint_out = run_stage_with_recovery(
      stage_num=stage_info["stage_num"],
      task_name=stage_info["task_name"],
      stage_description=stage_info["description"],
      iterations=stage_info["iterations"],
      num_envs=args.num_envs,
      initial_checkpoint=current_checkpoint,
      max_restarts=args.max_restarts,
      cooldown=args.cooldown,
      log_root=log_root,
    )

    if checkpoint_out:
      current_checkpoint = checkpoint_out
    else:
      print(f"\n[ARRET] Le curriculum a ete interrompu a l'etape {stage_info['stage_num']}.")
      sys.exit(1)

  print("\n" + "=" * 80)
  print("[SUCCES TOTAL] TOUTES LES ETAPES D'ENTRAINEMENT SONT TERMINEES AVEC SUCCES !")
  print(f"Dernier checkpoint disponible : {current_checkpoint}")
  print("=" * 80)
  print("\nPour visualiser et piloter votre robot au clavier, lancez simplement :")
  print("  python play_simple.py\n")


if __name__ == "__main__":
  main()
