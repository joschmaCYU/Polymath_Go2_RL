#!/usr/bin/env python3
"""Script de surveillance et auto-recovery pour l'entraînement RL de Unitree Go2.

En cas de plantage (crash CUDA, exception, erreur système), ce script détecte
automatiquement l'arrêt, trouve le dernier checkpoint disponible et relance
l'entraînement de manière sécurisée et robuste.

Sécurités intégrées :
- Arrêt automatique après 10 relances max (ou valeur configurée).
- Détection des crashs immédiats (démarrage/incompatibilité) pour stopper net toute boucle infinie.

Usage:
  python recover_train.py                           # Lance l'entraînement avec auto-recovery
  python recover_train.py --stage 1                 # Étape 1 : Terrain plat
  python recover_train.py --stage 2                 # Étape 2 : Escaliers et trous
  python recover_train.py --stage 3                 # Étape 3 : Big Map complexe
  python recover_train.py --num_envs 1024           # Préciser le nombre d'environnements
  python recover_train.py --max_restarts 10         # Nombre max de relances (défaut: 10)
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
  """Recherche le checkpoint model_*.pt le plus récent dans le dossier des logs."""
  if not log_dir.exists():
    return None
  checkpoints = list(log_dir.glob("**/model_*.pt"))
  if not checkpoints:
    return None
  return max(checkpoints, key=lambda p: p.stat().st_mtime)


def parse_args():
  parser = argparse.ArgumentParser(
    description="Superviseur auto-recovery pour l'entraînement Go2",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "--stage",
    "-s",
    type=str,
    default="3",
    choices=["1", "2", "3", "flat", "medium", "complex", "all"],
    help="Etape d'apprentissage: 1 (plat), 2 (escaliers et trous), 3 (big map complexe), all (enchaine les 3 etapes)",
  )
  parser.add_argument(
    "--task",
    "-t",
    type=str,
    default=None,
    help="Override direct du nom de la tâche (ex: Unitree-Go2-Complex, Unitree-Go2-Flat)",
  )
  parser.add_argument(
    "--num_envs",
    "-n",
    type=int,
    default=1024,
    help="Nombre d'environnements parallèles (ex: 512, 1024)",
  )
  parser.add_argument(
    "--max_iterations",
    "-i",
    type=int,
    default=5000,
    help="Nombre total d'itérations d'apprentissage",
  )
  parser.add_argument(
    "--save_interval",
    type=int,
    default=50,
    help="Intervalle de sauvegarde des checkpoints",
  )
  parser.add_argument(
    "--checkpoint",
    "-c",
    type=str,
    default=None,
    help="Checkpoint initial forcé pour commencer ou reprendre",
  )
  parser.add_argument(
    "--max_restarts",
    type=int,
    default=10,
    help="Nombre maximum de relances automatiques en cas de crash (défaut: 10)",
  )
  parser.add_argument(
    "--cooldown",
    type=float,
    default=3.0,
    help="Délai de pause en secondes avant une relance après crash",
  )
  return parser.parse_known_args()


def main():
  args, extra_args = parse_args()

  if args.stage == "all":
    train_all_script = Path(__file__).parent / "train_all.py"
    cmd = [sys.executable, str(train_all_script), "--num_envs", str(args.num_envs)]
    if args.checkpoint:
      cmd.extend(["--checkpoint", str(args.checkpoint)])
    if extra_args:
      cmd.extend(extra_args)
    sys.exit(subprocess.call(cmd))

  stage_to_task = {
    "1": "flat",
    "flat": "flat",
    "2": "stairs_holes",
    "medium": "stairs_holes",
    "3": "complex",
    "complex": "complex",
  }
  task_choice = args.task or stage_to_task.get(args.stage, "complex")

  log_root = Path("logs") / "rsl_rl" / "go2_velocity"
  restart_count = 0
  consecutive_rapid_failures = 0

  current_checkpoint = args.checkpoint
  if not current_checkpoint:
    latest = find_latest_checkpoint(log_root)
    if latest:
      print(f"[CHECKPOINT] Checkpoint existant detecte : {latest}")
      current_checkpoint = str(latest)

  print("=" * 75)
  print("[SUPERVISEUR] LANCEMENT DE L'ENTRAINEMENT AUTO-RECOVERY (GO2)")
  print(f"   * Tache/Etape      : {task_choice} (Stage {args.stage})")
  print(f"   * Envs             : {args.num_envs}")
  print(f"   * Max iterations   : {args.max_iterations}")
  print(f"   * Max relances     : {args.max_restarts}")
  if current_checkpoint:
    print(f"   * Point de depart  : {current_checkpoint}")
  else:
    print(f"   * Point de depart  : Nouvel entrainement a partir de zero")
  print("=" * 75)
  print("[INFO] Appuyez sur Ctrl+C a tout moment pour stopper l'entrainement.\n")

  interrupted = False

  while restart_count < args.max_restarts and not interrupted:
    cmd = [
      sys.executable,
      "train_simple.py",
      "--task",
      task_choice,
      "--num_envs",
      str(args.num_envs),
      "--max_iterations",
      str(args.max_iterations),
      "--save_interval",
      str(args.save_interval),
    ]

    if current_checkpoint:
      cmd.extend(["--checkpoint", str(current_checkpoint)])
    elif restart_count > 0:
      cmd.append("--resume")

    if extra_args:
      cmd.extend(extra_args)

    now_str = datetime.now().strftime("%H:%M:%S")
    if restart_count > 0:
      print(f"\n[{now_str}] [RECOVERY #{restart_count}/{args.max_restarts}] Relance de l'entrainement...")
      if current_checkpoint:
        print(f"[{now_str}] [REPRISE] Depuis le checkpoint : {current_checkpoint}")
    else:
      print(f"[{now_str}] [DEMARRAGE] Demarrage du processus d'entrainement...")

    start_time = time.time()

    try:
      proc = subprocess.Popen(cmd)

      def handle_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\n[ARRET] Demande par l'utilisateur. Fermeture propre du processus...")
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
        print("\n" + "=" * 75)
        print("[SUCCES] ENTRAINEMENT TERMINE AVEC SUCCES !")
        print("=" * 75)
        break
      else:
        if interrupted:
          break

        restart_count += 1
        now_str = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now_str}] [ATTENTION] Processus interrompu (Code: {exit_code}, duree: {duration:.1f}s).")

        # Securite anti-boucle : crash rapide (< 30 secondes)
        if duration < 30.0:
          consecutive_rapid_failures += 1
          print(f"[{now_str}] [ATTENTION] Detection d'un crash precoce ({consecutive_rapid_failures}/2 consecutifs).")
          if consecutive_rapid_failures >= 2:
            print("\n" + "!" * 75)
            print("[ERREUR CRITIQUE] Crash repete au demarrage (duree < 30s).")
            print("   La relance automatique est stoppee pour eviter une boucle infinie.")
            print("   Verifiez les logs d'erreur ci-dessus (ex: memoire GPU insuffisante ou parametre invalide).")
            print("!" * 75)
            sys.exit(1)
        else:
          # Si le processus a tourne plus de 30 secondes, reinitialiser le compteur
          consecutive_rapid_failures = 0

        # Verifier le plafond de relances
        if restart_count >= args.max_restarts:
          print(f"\n[ERREUR] Plafond maximal de {args.max_restarts} relances atteint. Arret du superviseur.")
          sys.exit(1)

        # Rechercher le dernier checkpoint valide
        latest = find_latest_checkpoint(log_root)
        if latest:
          current_checkpoint = str(latest)
          print(f"[{now_str}] [CHECKPOINT] Dernier checkpoint sauvegarde : {current_checkpoint}")
        else:
          print(f"[{now_str}] [INFO] Aucun checkpoint trouve, reprise depuis le debut.")

        print(f"[PAUSE] Pause de {args.cooldown}s pour liberer les ressources GPU...")
        time.sleep(args.cooldown)

    except KeyboardInterrupt:
      interrupted = True
      print("\n[FIN] Arret du superviseur.")
      break


if __name__ == "__main__":
  main()
