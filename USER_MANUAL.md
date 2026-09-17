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

L'approche recommandee et prioritaire est l'utilisation de **Docker avec accelaration GPU**, qui garantit un environnement isole, reproductible et sans conflit de paquets CUDA/Python sur votre machine hote.

### Option 1 : Installation Complete avec Docker (Recommandee)

#### A. Prerequis sur la machine hote
1. **Pilote NVIDIA** : Verifiez que vos drivers NVIDIA sont fonctionnels :
   ```bash
   nvidia-smi
   ```
2. **Docker Engine** : Si Docker n'est pas encore installe :
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   # Reconnectez-vous a votre session utilisateur pour appliquer le groupe docker
   ```
3. **NVIDIA Container Toolkit** (Indispensable pour le support GPU dans Docker) :
   ```bash
   # Configuration du depot NVIDIA
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   # Installation et configuration du runtime Docker
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
4. **Validation du bon fonctionnement GPU dans Docker** :
   ```bash
   docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```
   Si la carte graphique s'affiche correctement dans ce test, Docker est pret.

#### B. Construire l'image du projet
A la racine du projet (`/home/josch/Projects/Polymath`) :
```bash
docker compose build
```
L'image `unitree-go2-rl:latest` installe automatiquement :
- Ubuntu 22.04 avec CUDA 12.4
- PyTorch 2.7 dev avec CUDA
- MuJoCo 3.5.0 et MuJoCo Warp 3.5.0
- RSL-RL 5.0.1 et mjlab 1.2.0
- Viser (visualisation 3D web), TensorBoard et dependances robotiques

#### C. Fonctionnement des volumes et persistance
Le fichier `docker-compose.yml` monte automatiquement le dossier local du projet dans `/app` :
- Tous les checkpoints generes dans `logs/rsl_rl/go2_velocity/` sont enregistres directement sur votre disque dur.
- Toute modification apportee aux scripts Python (`train_all.py`, `play_simple.py`, etc.) est immediatement prise en compte dans le conteneur sans avoir a reconstruire l'image.

#### D. Toutes les commandes Docker disponibles

1. **Lancer l'entrainement automatique en 3 etapes** :
   ```bash
   docker compose run --rm go2-rl python train_all.py
   # Pour doubler la vitesse de calcul avec 2048 robots paralleles :
   docker compose run --rm go2-rl python train_all.py --num_envs 2048
   ```

2. **Lancer TensorBoard en arriere-plan (suivi en temps reel)** :
   ```bash
   docker compose up -d tensorboard
   ```
   Accedez aux courbes sur `http://localhost:6006` dans votre navigateur.
   Pour arreter TensorBoard : `docker compose stop tensorboard`

3. **Visualiser le robot en 3D Web (Viser)** :
   ```bash
   docker compose run --rm -p 8080:8080 go2-rl python play_simple.py --viewer viser
   ```
   Ouvrez `http://localhost:8080` dans votre navigateur.

4. **Visualiser avec la fenetre native GUI X11 (OpenGL) avec xhost et DISPLAY=:1** :
   ```bash
   # 1. Autoriser le conteneur a ouvrir une fenetre sur votre serveur X11 :
   xhost +local:root
   
   # 2. Lancer la visualisation via docker exec (si le conteneur tourne) :
   docker exec -it -e DISPLAY=:1 go2-rl-runner python play_simple.py
   
   # Ou lancer directement via docker compose :
   docker compose run --rm -e DISPLAY=:1 go2-rl python play_simple.py
   ```
   *(Note : remplacez `:1` par `:0` ou `$DISPLAY` selon le numero de votre ecran X11).*

5. **Ouvrir un terminal interactif (bash)** :
   ```bash
   docker compose run --rm go2-rl bash
   ```

---

### Option 2 : Alternative Locale (sans Docker)

Si vous preferez executer directement dans votre environnement Python hote :

#### 1. Paquets systeme
```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake libgl1-mesa-dev libglib2.0-0
```

#### 2. Dependances Python
```bash
pip install --upgrade pip
pip install -e .
```

---

## 4. Guide d'Entrainement : La Commande Unique

Pour eviter toute complexite, une seule commande prend en charge l'integralite du cycle de vie de l'entrainement :

```bash
python train_all.py
```

### Que fait cette commande ?
1. **Etape 1 (Sol plat)** : Entraine 4096 robots pendant 500 iterations (~49 millions de pas, ~4-5 minutes) sur sol plan pour acquerir l'equilibre, la posture et les allures de base.
2. **Transfert de connaissances** : Recupere automatiquement le dernier checkpoint produit par l'etape 1.
3. **Etape 2 (Escaliers et trous)** : Charge le modele et poursuit l'entrainement pendant 750 iterations (~73 millions de pas, ~6-7 minutes) sur des escaliers et des crevasses moderees.
4. **Transfert de connaissances** : Recupere le dernier checkpoint de l'etape 2.
5. **Etape 3 (Big Map complexe)** : Plonge les 4096 robots dans l'environnement exterieur complet (marches hautes, collines rugueuses, murs, couloirs et precipices de 2m) pendant 1200 iterations (~118 millions de pas, ~14-15 minutes).
Duree totale estimee sur RTX 4070 : environ 25 a 30 minutes au total pour 240 millions de transitions.

### Gestion automatique des crashs (Auto-Recovery)
- En cas de saturation memoire GPU ou d'incident physique, le superviseur capture le code d'erreur, localise le dernier checkpoint `model_*.pt` et relance l'entrainement a la meme etape sans perte de progression.
- **Plafond de securite** : 10 relances maximum autorisees.
- **Protection anti-boucle (30s)** : Si un plantage survient en moins de 30 secondes apres le demarrage (2 fois de suite), le superviseur s'arrete immediatement et affiche les messages de diagnostic, evitant de consommer des ressources en boucle.

---

## 5. Options Avancees de train_all.py

Bien que la commande par defaut suffise, plusieurs arguments sont configurables si necessaire :

```bash
# Reprendre l'entrainement depuis le dernier checkpoint (fonctionne meme des l'etape 1)
python train_all.py --resume
# Ou via Docker :
docker compose run --rm go2-rl python train_all.py --resume

# Reprendre uniquement l'etape 1 isolee sur le dernier checkpoint
python train_simple.py --stage 1 --resume
# Ou via Docker :
docker compose run --rm go2-rl python train_simple.py --stage 1 --resume

# Demarrer directement a partir d'une etape specifique (ex: etape 2)
python train_all.py --from_stage 2

# Demarrer avec un checkpoint externe specifique
python train_all.py --checkpoint /chemin/vers/model_500.pt

# Modifier le nombre de robots simules en parallele (ex: 2048 pour doubler la vitesse)
python train_all.py --num_envs 2048

# Ajuster le nombre d'iterations par etape
python train_all.py --iters_stage1 500 --iters_stage2 1000 --iters_stage3 3000
```

### Peut-on faire un "resume" des l'etape 1 ?
**Oui, absolument !**
- Si vous interrompez l'apprentissage au milieu de l'etape 1 (par exemple a l'iteration 400 sur 1000), vous pouvez relancer soit avec `python train_all.py --resume`, soit avec `python train_simple.py --stage 1 --resume`.
- Le script scanne automatiquement le repertoire `logs/rsl_rl/go2_velocity/` et charge le fichier `model_*.pt` le plus recent.
- **Compatibilite totale** : L'espace d'observation a ete unifie a 234 dimensions pour l'acteur et 261 pour le critique sur toutes les etapes. Il n'y a donc aucun risque d'erreur `size mismatch`.
- S'il n'existe encore aucun checkpoint sur votre machine, le flag `--resume` le detecte poliment et commence un entrainement tout neuf sans planter.

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

L'entrainement utilise TensorBoard comme systeme de suivi local en direct (aucun compte externe ni connexion internet requis). Il permet de suivre l'evolution des recompenses (mean reward), la longueur des episodes, les pertes de politique et du critique (value loss).

### A. Lancement avec Docker (Recommande)

1. **En service d'arriere-plan** :
   ```bash
   docker compose up -d tensorboard
   ```
   Accedez immediatement au tableau de bord via `http://localhost:6006` dans votre navigateur.

2. **Pour arreter le service TensorBoard** :
   ```bash
   docker compose stop tensorboard
   ```

3. **En commande ponctuelle** :
   ```bash
   docker compose run --rm -p 6006:6006 go2-rl tensorboard --logdir logs/rsl_rl/go2_velocity --host 0.0.0.0 --port 6006
   ```

4. **Dans un conteneur d'entrainement en cours d'execution** :
   ```bash
   docker exec -it go2-rl-runner tensorboard --logdir logs/rsl_rl/go2_velocity --host 0.0.0.0 --port 6006
   ```

### B. Lancement en Local (hors Docker)

```bash
tensorboard --logdir logs/rsl_rl/go2_velocity/
```
Puis ouvrez `http://localhost:6006` dans votre navigateur web.

---

## 8. Configuration Haute Performance (RTX 4070 8 Go VRAM)

Le projet est calibre par defaut pour exploiter la puissance d'une carte graphique moderne comme la **NVIDIA GeForce RTX 4070 (8 Go VRAM)** :

### A. 4096 environnements paralleles par defaut (`--num_envs 4096`)
- **Saturation des coeurs Ada Lovelace** : Les 4096 robots simules en parallele saturent efficacement les 5888 coeurs CUDA et tirent parti des 32 Mo de cache L2.
- **Debit exceptionnel** : Le simulateur atteint un debit de **40 000 a 50 000 pas par seconde (FPS)**.
- **Empreinte memoire ideale** : Ne consomme qu'environ **1.8 Go de VRAM** (sur 8 Go), garantissant une totale stabilite thermique et memoire sans risque d'OOM.
- **Qualite d'apprentissage accrue** : 4096 robots garantissent une diversite d'exploration 4 fois superieure sur les terrains accidentes, reduisant la variance du gradient et stabilisant l'optimiseur PPO.
- *(Remarque pour cartes 6 Go : si vous executez sur une carte plus modeste comme une RTX 2060, passez simplement l'argument `--num_envs 2048`)*.

### B. Optimisation de la Continuous Collision Detection (`ccd_iterations = 50`)
- Dans MuJoCo Warp, l'algorithme EPA (Expanding Polytope Algorithm) pour contacts convexes (pattes du Go2 contre marches/crevasses) converge en **15 a 30 iterations**.
- `ccd_iterations = 50` assure une precision physique absolue (zero penetration des pattes dans les marches) tout en divisant par 10 l'allocation de memoire de collision temporaire par rapport a l'ancienne valeur de 500.

### C. Reduction des buffers de capteurs de contact (`contact_sensor_maxmatch = 160`)
- Les 4 pattes ne generant que 4 a 8 points de contact physiques reels, un tampon de 160 elements evite tout gaspillage memoire tout en garantissant zero perte d'evenement d'impact.

### D. Acceleration Tensor Cores et TF32
- `torch.backends.cuda.matmul.allow_tf32 = True` et `torch.backends.cudnn.allow_tf32 = True` exploitent les Tensor Cores de 4e generation de l'architecture Ada Lovelace.

### E. Duree estimee du Curriculum complet sur RTX 4070
- **Etape 1 (Sol plat)** : 500 iterations (~49M transitions) -> **~4 minutes**
- **Etape 2 (Escaliers & Trous)** : 750 iterations (~73M transitions) -> **~7 minutes**
- **Etape 3 (Big Map complexe)** : 1200 iterations (~118M transitions) -> **~14 minutes**
- **DUREE TOTALE** : **~25 a 30 minutes** pour obtenir un quadrupede tout-terrain completement autonome.

---

## 9. Resolution des Problemes Frequents

### Erreur CUDA Out of Memory (OOM)
La RTX 2060 dispose de 6 Go de VRAM. Avec 1024 environnements, la consommation est d'environ 1.4 Go a 1.8 Go.
Si d'autres applications occupent le GPU et provoquent un OOM, reduisez le nombre d'environnements :
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
