# Cheat Sheet - Unitree Go2 Reinforcement Learning

Ce guide recapitulatif regroupe toutes les commandes essentielles pour l'entrainement, la visualisation et l'execution Docker du robot Unitree Go2 sur terrains exterieurs complexes.

---

## 1. La Commande Unique (Recommandee)

Pour lancer l'entrainement complet calibre pour RTX 4070 (8 Go VRAM) qui passe automatiquement par les 3 etapes en ~25 a 30 minutes :

```bash
docker compose run --rm go2-rl python train_all.py
# Ou en local :
python train_all.py
```

Cette commande execute sequentiellement avec 4096 robots paralleles :
1. Etape 1 : Sol plat (500 iterations = ~49 millions de pas, ~4-5 min)
2. Etape 2 : Escaliers et trous (750 iterations = ~73 millions de pas, ~6-7 min)
3. Etape 3 : Big Map complexe (1200 iterations = ~118 millions de pas, ~14-15 min)

Chaque etape surveille les erreurs GPU, redemarre automatiquement au dernier checkpoint en cas de plantage (jusqu'a 10 fois), et s'arrete proprement si un crash se produit en moins de 30 secondes.

---

| Action | Commande | Description |
| :--- | :--- | :--- |
| **Entrainement complet (Docker)** | `docker compose run --rm go2-rl python train_all.py` | Enchaine les 3 etapes dans le conteneur GPU isole |
| **Entrainement complet (Local)** | `python train_all.py` | Enchaine Plat -> Escaliers -> Big Map avec auto-recovery |
| **Suivi TensorBoard (Docker)** | `docker compose up -d tensorboard` | Tableau de bord courbes RL ouvert sur `http://localhost:6006` |
| **Visualisation Web 3D (Docker)** | `docker compose run --rm -p 8080:8080 go2-rl python play_simple.py --viewer viser` | Visualiseur 3D interactif dans le navigateur `http://localhost:8080` |
| **Visualisation GUI Native (Docker)** | `docker exec -it -e DISPLAY=:1 go2-rl-runner python play_simple.py` | Affichage graphique natif OpenGL via X11 / xhost |
| **Visualisation au clavier (Local)** | `python play_simple.py` | Charge le dernier modele et permet de piloter aux fleches |
| **Entrainement supervise par etape** | `python recover_train.py --stage 1` | Entraine uniquement l'etape 1 avec auto-recovery (30s) |
| **Entrainement simple manuel** | `python train_simple.py --stage 1` | Lance un entrainement direct sans superviseur |
| **Shell interactif Docker** | `docker compose run --rm go2-rl bash` | Ouvre un terminal bash directement a l'interieur du conteneur |

---

## 3. Options de l'Entrainement Automatique (`train_all.py`)

```bash
# Lancement standard de zero (1024 robots paralleles)
python train_all.py

# Reprendre depuis le dernier checkpoint sauvegarde (fonctionne meme des l'etape 1)
python train_all.py --resume
# Ou via Docker :
docker compose run --rm go2-rl python train_all.py --resume

# Commencer directement a l'etape 2 ou 3
python train_all.py --from_stage 2
python train_all.py --from_stage 3

# Reprendre uniquement l'etape 1 isolee sur le dernier checkpoint
python train_simple.py --stage 1 --resume
# Ou via Docker :
docker compose run --rm go2-rl python train_simple.py --stage 1 --resume

# Reprendre avec un checkpoint specifique
python train_all.py --checkpoint logs/rsl_rl/go2_velocity/2026-09-17_19-34-26_flat/model_500.pt

# Ajuster le nombre de robots simules (ex: 2048 pour doubler la vitesse sur RTX 2060)
python train_all.py --num_envs 2048

# Personnaliser le nombre d'iterations par etape
python train_all.py --iters_stage1 500 --iters_stage2 1000 --iters_stage3 2000
```

---

## 4. Visualisation et Pilotage au Clavier (`play_simple.py`)

Une fois le modele entraine, visualisez le comportement du robot :

```bash
# Charge automatiquement le tout dernier checkpoint sur la Big Map
python play_simple.py

# Visualiser sur l'etape 1 (plat) ou l'etape 2 (escaliers et trous)
python play_simple.py --stage 1
python play_simple.py --stage 2

# Forcer un checkpoint specifique
python play_simple.py --checkpoint logs/rsl_rl/go2_velocity/2026-09-17_19-00-00_complex/model_2500.pt

# Affichage Web (si vous etes sur serveur distant ou sans ecran local)
python play_simple.py --viewer viser
```

### Touches de controle (cliquez sur la fenetre 3D pour donner le focus) :
- Fleche HAUT ou W : Avancer
- Fleche BAS ou S : Reculer
- Fleche GAUCHE ou A : Tourner a gauche (rotation yaw)
- Fleche DROITE ou D : Tourner a droite (rotation yaw)
- Espace ou X : Arret immediat (vitesse zero)
- Entree : Reinitialiser le robot a sa position de depart

---

## 5. Commandes Docker (Serveurs distants ou GPU dedie)

### 1. Construire l'image Docker
Sur la branche `cuda-12-1` (calibree pour les pilotes NVIDIA 535.x et CUDA 12.1) :
```bash
docker compose build
```

### 2. Lancer l'entrainement complet dans Docker
```bash
docker compose run --rm go2-rl python train_all.py
```

### 3. Lancer la visualisation Web 3D depuis Docker
```bash
docker compose run --rm -p 8080:8080 go2-rl python play_simple.py --viewer viser
```
Ouvrez ensuite `http://localhost:8080` dans votre navigateur Web.

### 4. Visualisation GUI Native (X11 / OpenGL) avec xhost et DISPLAY=:1

Pour afficher la fenetre 3D native de MuJoCo directement sur votre ecran X11 depuis Docker :

1. Autoriser l'acces local au serveur X11 sur votre machine hote :
```bash
xhost +local:root
# Ou pour autoriser toutes les connexions locales :
xhost +
```

2. Executer dans un conteneur deja en cours d'execution (`docker exec`) :
```bash
docker exec -it -e DISPLAY=:1 go2-rl-runner python play_simple.py
```
*(Remplacez `:1` par `:0` ou `$DISPLAY` si votre serveur graphique tourne sur l'ecran 0)*

3. Ou lancer un nouveau conteneur ponctuel avec affichage :
```bash
docker compose run --rm -e DISPLAY=:1 go2-rl python play_simple.py
```

4. Equivalent direct avec la commande pure `docker run` :
```bash
docker run --rm -it \
  --runtime=nvidia \
  --ipc=host \
  --net=host \
  -e DISPLAY=:1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/app \
  unitree-go2-rl:latest python play_simple.py
```

### 5. Lancer TensorBoard dans Docker

Visualisez en direct les courbes d'apprentissage, recompenses et pertes de politique sans surcharger votre machine :

1. En arriere-plan via docker-compose (recommande) :
```bash
docker compose up -d tensorboard
```
Accedez ensuite a l'interface web : `http://localhost:6006`

Pour arreter TensorBoard :
```bash
docker compose stop tensorboard
```

2. En commande directe ponctuelle :
```bash
docker compose run --rm -p 6006:6006 go2-rl tensorboard --logdir logs/rsl_rl/go2_velocity --host 0.0.0.0 --port 6006
```

3. Dans un conteneur deja en cours d'execution (`docker exec`) :
```bash
docker exec -it go2-rl-runner tensorboard --logdir logs/rsl_rl/go2_velocity --host 0.0.0.0 --port 6006
```

4. Equivalent direct avec pure `docker run` :
```bash
docker run --rm -it -p 6006:6006 -v $(pwd):/app unitree-go2-rl:latest tensorboard --logdir logs/rsl_rl/go2_velocity --host 0.0.0.0 --port 6006
```

### 6. Ouvrir un Terminal Interactif dans le Conteneur Docker

Pour tester des commandes Python, explorer l'arborescence ou executer des scripts arbitraires :
```bash
docker compose run --rm go2-rl bash
# Ou si le conteneur go2-rl-runner tourne deja :
docker exec -it go2-rl-runner bash
```

---

## 6. Configuration Haute Performance (RTX 4070 8 Go)

L'environnement est calibre par defaut pour exploiter la puissance d'une RTX 4070 et terminer l'apprentissage en ~25 a 30 minutes :

1. **4096 robots simules en parallele (`--num_envs 4096`)** :
   - Exploite a fond les coeurs CUDA et le large cache L2 de l'architecture Ada Lovelace.
   - Debit atteignant **40 000 a 50 000 steps/seconde**.
   - VRAM occupee : ~1.8 Go sur les 8 Go disponibles (large marge de securite).
   - *(Note : si vous lancez temporairement sur une carte 6 Go comme une RTX 2060, passez simplement `--num_envs 2048`)*.

2. **Detection de collision continue optimisee (`ccd_iterations = 50`)** :
   - L'algorithme de collision convexe EPA converge en 20-30 iterations pour les pattes du Go2.
   - La valeur 50 garantit zero penetration sur les marches tout en eliminant le tampon temporaire de 1.31 Go alloue par l'ancienne valeur de 500.

3. **Curriculum en 2450 iterations totales** :
   - Etape 1 (Plat) : 500 iters (~49M pas, ~4 min)
   - Etape 2 (Escaliers/Trous) : 750 iters (~73M pas, ~7 min)
   - Etape 3 (Big Map complexe) : 1200 iters (~118M pas, ~14 min)
   - Plus de 240 millions de pas collectes au total pour une politique tout-terrain extremement robuste.

4. **Suivi local sans latence reseau avec TensorBoard** :
   - L'utilisation de TensorBoard (`docker compose up -d tensorboard`) evite les ralentissements lies a la synchronisation internet de services externes comme wandb.

---

## 7. Structure des Fichiers

- train_all.py : Pipeline d'apprentissage complet en 3 etapes avec auto-recovery et securite 30s
- play_simple.py : Visualisation 3D avec pilotage interactif au clavier
- recover_train.py : Superviseur d'entrainement avec relance automatique sur crash (>30s)
- train_simple.py : Script de base d'entrainement MuJoCo / RSL-RL
- CHEAT_SHEET.md : Ce guide de reference
- USER_MANUAL.md : Manuel complet d'architecture et de fonctionnement
- Dockerfile / docker-compose.yml : Configuration pour execution conteneurisee
- src/tasks/velocity/terrains.py : Generateur de terrains complexes (marches, collines, murs, couloirs, fosses)
- src/tasks/velocity/config/go2/ : Configurations d'apprentissage Go2
