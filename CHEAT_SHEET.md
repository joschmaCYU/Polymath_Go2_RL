# Cheat Sheet - Unitree Go2 Reinforcement Learning

Ce guide recapitulatif regroupe toutes les commandes essentielles pour l'entrainement, la visualisation et l'execution Docker du robot Unitree Go2 sur terrains exterieurs complexes.

---

## 1. La Commande Unique (Recommandee)

Pour lancer l'entrainement complet qui passe automatiquement par les 3 etapes avec gestion des crashs et reprise automatique des checkpoints :

```bash
python train_all.py
```

Cette commande execute sequentiellement :
1. Etape 1 : Sol plat (1000 iterations)
2. Etape 2 : Escaliers et trous (1500 iterations) a partir du checkpoint de l'etape 1
3. Etape 3 : Big Map complexe (2500 iterations) a partir du checkpoint de l'etape 2

Chaque etape surveille les erreurs GPU, redemarre automatiquement au dernier checkpoint en cas de plantage (jusqu'a 10 fois), et s'arrete proprement si un crash se produit en moins de 30 secondes.

---

## 2. Tableau Recapitulatif des Commandes

| Action | Commande | Description |
| :--- | :--- | :--- |
| **Entrainement complet (3 etapes)** | `python train_all.py` | Enchaine Plat -> Escaliers -> Big Map avec auto-recovery |
| **Visualisation au clavier** | `python play_simple.py` | Charge le dernier modele et permet de piloter aux fleches |
| **Visualisation Web 3D** | `python play_simple.py --viewer viser` | Visualiseur 3D interactif dans votre navigateur Web |
| **Entrainement supervise par etape** | `python recover_train.py --stage 1` | Entraine uniquement l'etape 1 avec auto-recovery (30s) |
| **Entrainement simple manuel** | `python train_simple.py --stage 1` | Lance un entrainement direct sans superviseur |

---

## 3. Options de l'Entrainement Automatique (`train_all.py`)

```bash
# Lancement standard avec 1024 robots paralleles (recommande pour 6 Go VRAM)
python train_all.py

# Commencer directement a l'etape 2 ou 3
python train_all.py --from_stage 2
python train_all.py --from_stage 3

# Ajuster le nombre de robots simules
python train_all.py --num_envs 512

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

---

## 6. Structure des Fichiers

- train_all.py : Pipeline d'apprentissage complet en 3 etapes avec auto-recovery et securite 30s
- play_simple.py : Visualisation 3D avec pilotage interactif au clavier
- recover_train.py : Superviseur d'entrainement avec relance automatique sur crash (>30s)
- train_simple.py : Script de base d'entrainement MuJoCo / RSL-RL
- CHEAT_SHEET.md : Ce guide de reference
- USER_MANUAL.md : Manuel complet d'architecture et de fonctionnement
- Dockerfile / docker-compose.yml : Configuration pour execution conteneurisee
- src/tasks/velocity/terrains.py : Generateur de terrains complexes (marches, collines, murs, couloirs, fosses)
- src/tasks/velocity/config/go2/ : Configurations d'apprentissage Go2
