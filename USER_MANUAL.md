# Manuel Utilisateur - Reinforcement Learning pour Unitree Go2

Ce document fournit un guide complet pour l'installation, la configuration, la comprehension de l'architecture du code et l'execution des entrainements pour le robot quadrupede Unitree Go2 sur terrains complexes.

---

## 1. Vue d'Ensemble du Projet

Ce projet a pour objectif d'entrainer un robot quadrupede Unitree Go2 a naviguer de facon autonome sur des terrains exterieurs difficiles et accidentes en utilisant l'apprentissage par renforcement (Deep Reinforcement Learning, PPO).

### Technologies utilisees
- Moteur physique : MuJoCo 3.5 et MuJoCo Warp (GPU-accelerated physics).
- Algorithme d'apprentissage : RSL-RL (PPO - Proximal Policy Optimization).
- Calcul et reseaux neuronaux : PyTorch avec CUDA.
- Visualisation 3D : MuJoCo Native Viewer (OpenGL) ou Viser (interface 3D accessible dans le navigateur web).

---

## 2. Architecture Globale du Code

Le systeme repose sur une boucle fermee entre la simulation physique sur GPU et le reseau de neurones PPO.

### A. Reseaux de neurones et Dimensions d'Observation
Pour permettre un passage fluide entre les differents niveaux de difficulte (Curriculum Learning) sans erreur de compatibilite de dimension (size mismatch), l'espace d'observation est strictement unifie :
- Acteur (Actor) : 234 dimensions
  - Vitesse angulaire de la base (3 dimensions)
  - Vecteur gravite projete (3 dimensions)
  - Consignes de vitesse utilisateur vx, vy, yaw (3 dimensions)
  - Positions articulaires actuelles des 12 moteurs (12 dimensions)
  - Vitesses articulaires des 12 moteurs (12 dimensions)
  - Dernieres actions appliquees (12 dimensions)
  - Balayage du relief du terrain (Height scan) : grille de 187 points d'altitude mesures autour du robot.
- Critique (Critic) : 261 dimensions (observations de l'acteur + etats privilegies comme la vitesse lineaire exacte du robot).

Grace a cette standardisation, n'importe quel checkpoint sauvegarde a l'etape 1 peut etre directement charge par l'etape 2 ou l'etape 3 sans aucune modification architecturale.

### B. Actionneurs du Robot
Le robot possede 12 degres de liberte (3 moteurs par patte : hanche avant/arriere, hanche laterale, genou).
Le reseau predit des positions articulaires cibles (PD control) qui sont ensuite converties en couples moteurs par les regulateurs de MuJoCo.

### C. Generateur de Terrains (src/tasks/velocity/terrains.py)
Le terrain de simulation est genere de facon procedurale sous forme de sous-terrains imbriques :
1. Sol plat : zone plane sans obstacle pour le rodage de la posture.
2. Escaliers montants et descendants : marches de 10 a 20 cm.
3. Marches inversees et irregulieres : obstacles requiring foot clearance.
4. Collines accidentes : relief de type bruit de Perlin et bosses pyramidales.
5. Murs : blocs verticaux bloquants.
6. Longs couloirs : corridors etroits bordes de murs.
7. Trous et fosses profondes : precipices de 2 metres de profondeur avec plateformes espacées (gap terrain et stepping stones).

---

## 3. Installation et Configuration

### Option 1 : Installation Locale (Linux / Ubuntu)

#### 1. Verifier les prerequis systeme
- Python 3.10, 3.11 ou 3.12
- Pilote NVIDIA avec CUDA (ex: CUDA 12.x ou 13.x)
- Carte graphique NVIDIA (minimum 6 Go de VRAM, par exemple RTX 2060)

#### 2. Installer les paquets de base
```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake libgl1-mesa-dev libglib2.0-0
```

#### 3. Installer les dependances Python
Dans la racine du projet :
```bash
pip install --upgrade pip
pip install -e .
```
Les versions cles verifiees et figees sont :
- mjlab == 1.2.0
- mujoco == 3.5.0
- mujoco-warp == 3.5.0
- warp-lang == 1.12.0
- rsl-rl-lib == 5.0.1
- torch == 2.7.0.dev20250220+cu124 (ou version compatible CUDA 12.x)

---

### Option 2 : Installation Conteneurisee avec Docker

Si vous travaillez sur un serveur distant ou souhaitez une isolation complete :

#### 1. Construire l'image Docker
```bash
docker compose build
```

#### 2. Lancer l'entrainement dans le conteneur
```bash
docker compose run --rm go2-rl python train_all.py
```

#### 3. Lancer la visualisation Web 3D depuis Docker
```bash
docker compose run --rm -p 8080:8080 go2-rl python play_simple.py --viewer viser
```
Puis ouvrez `http://localhost:8080` dans votre navigateur.

---

## 4. Guide d'Entrainement : La Commande Unique

Pour eviter toute complexite, une seule commande prend en charge l'integralite du cycle de vie de l'entrainement :

```bash
python train_all.py
```

### Que fait cette commande ?
1. **Etape 1 (Sol plat)** : Entraine le robot pendant 1000 iterations sur sol plan pour acquerir l'equilibre, la posture et les allures de base.
2. **Transfert de connaissances** : Recupere automatiquement le dernier checkpoint produit par l'etape 1.
3. **Etape 2 (Escaliers et trous)** : Charge le modele et poursuit l'entrainement pendant 1500 iterations sur des escaliers et des crevasses moderees.
4. **Transfert de connaissances** : Recupere le dernier checkpoint de l'etape 2.
5. **Etape 3 (Big Map complexe)** : Plonge le robot dans l'environnement exterieur complet (marches hautes, collines rugueuses, murs, couloirs et precipices de 2m) pendant 2500 iterations.

### Gestion automatique des crashs (Auto-Recovery)
- En cas de saturation memoire GPU ou d'incident physique, le superviseur capture le code d'erreur, localise le dernier checkpoint `model_*.pt` et relance l'entrainement a la meme etape sans perte de progression.
- **Plafond de securite** : 10 relances maximum autorisees.
- **Protection anti-boucle (30s)** : Si un plantage survient en moins de 30 secondes apres le demarrage (2 fois de suite), le superviseur s'arrete immediatement et affiche les messages de diagnostic, evitant de consommer des ressources en boucle.

---

## 5. Options Avancees de train_all.py

Bien que la commande par defaut suffise, plusieurs arguments sont configurables si necessaire :

```bash
# Modifier le nombre de robots simules en parallele (defaut: 1024)
python train_all.py --num_envs 512

# Demarrer directement a partir d'une etape specifique (ex: etape 2)
python train_all.py --from_stage 2

# Ajuster le nombre d'iterations par etape
python train_all.py --iters_stage1 500 --iters_stage2 1000 --iters_stage3 3000

# Demarrer avec un checkpoint externe specifique
python train_all.py --checkpoint /chemin/vers/model_500.pt
```

---

## 6. Visualisation et Pilotage au Clavier

Apres l'entrainement, vous pouvez verifier les performances de votre modele et le diriger manuellement :

```bash
# Charge automatiquement le tout dernier modele entraine sur la Big Map
python play_simple.py

# Tester le modele sur le terrain d'etape 1 ou 2
python play_simple.py --stage 1
python play_simple.py --stage 2

# Choisir le visualiseur web Viser
python play_simple.py --viewer viser
```

### Controles Clavier
Cliquez sur la fenetre de rendu pour lui donner le focus :
- Fleche HAUT (ou W) : Faire avancer le robot (incremente vx jusqu'a 2.0 m/s)
- Fleche BAS (ou S) : Faire reculer le robot (jusqu'a -1.0 m/s)
- Fleche GAUCHE (ou A) : Rotation yaw vers la gauche (jusqu'a 1.8 rad/s)
- Fleche DROITE (ou D) : Rotation yaw vers la droite (jusqu'a -1.8 rad/s)
- Barre d'espace (ou X) : Arret complet immediat (vx = 0, wz = 0)
- Entree : Reinitialiser le robot (reset simulation)

---

## 7. Suivi des Metriques avec TensorBoard

L'entrainement utilise TensorBoard comme systeme de suivi local (aucun compte externe ni connexion internet requis).

Pour visualiser les courbes de recompense, les pertes de politique et la longueur des episodes :
```bash
tensorboard --logdir logs/rsl_rl/go2_velocity/
```
Puis ouvrez `http://localhost:6006` dans votre navigateur web.

---

## 8. Resolution des Problemes Frequents

### Erreur CUDA Out of Memory (OOM)
La RTX 2060 dispose de 6 Go de VRAM. Avec 1024 environnements, la consommation est d'environ 1.4 Go a 1.8 Go.
Si d'autres applications utilisent le GPU, reduisez le nombre d'environnements :
```bash
python train_all.py --num_envs 512
```

### Le superviseur s'arrete avec l'erreur "Crash repete au demarrage (duree < 30s)"
Cette securite se declenche si le processus plante prematurement. Verifiez :
1. Que votre pilote NVIDIA fonctionne correctement (`nvidia-smi`).
2. Qu'aucun processus fantome n'occupe la memoire GPU (`killall -9 python`).
3. Que le chemin de checkpoint fourni existe bien sur votre disque.

### Le robot tombe souvent dans les trous a l'etape 3
Le saut et le contournement de gouffres de 2 metres necessitent un temps d'exploration substantiel. Augmentez le nombre d'iterations de l'etape 3 :
```bash
python train_all.py --from_stage 3 --iters_stage3 5000
```
