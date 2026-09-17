# Unitree Go2 Reinforcement Learning (MuJoCo Warp / mjlab)

Framework d'apprentissage par renforcement léger, optimisé et épuré pour le robot quadrupède **Unitree Go2**, entraîné sur des **terrains extérieurs complexes** (marches, terrains accidentés, murs, longs couloirs, trous).

Ce projet a été nettoyé pour ne conserver que les composants indispensables au Go2.

---

## 1. Demarrage Rapide

### 1. Entrainer le Go2 en 3 etapes automatiques (Recommande)
```bash
# Enchaine automatiquement Etape 1 (plat) -> Etape 2 (escaliers/trous) -> Etape 3 (Big Map)
python train_all.py
```

### 2. Jouer / Piloter le Go2 au clavier
```bash
# Charge automatiquement le dernier checkpoint entraine
python play_simple.py
```
> **Controles clavier (cliquez dans la fenetre graphique pour focus) :**
> - **Fleche HAUT / W** : Avancer
> - **Fleche BAS / S** : Reculer
> - **Fleche GAUCHE / A** : Tourner a gauche
> - **Fleche DROITE / D** : Tourner a droite
> - **Espace / X** : Arret complet

---

## 2. Curriculum en 3 Etapes

Pour un entrainement progressif etape par etape :

1. **Etape 1 : Sol Plat**
   ```bash
   python train_simple.py --stage 1 --max_iterations 1500 --num_envs 1024
   ```
2. **Etape 2 : Escaliers et Trous**
   ```bash
   python train_simple.py --stage 2 --resume --max_iterations 3000 --num_envs 1024
   ```
3. **Etape 3 : Big Map Exterieure Complexe**
   ```bash
   python train_simple.py --stage 3 --resume --max_iterations 6000 --num_envs 1024
   ```

---

## 3. Superviseur Anti-Crash (recover_train.py)

Gere les aleas materiels ou plantages GPU :
```bash
# Surveille le processus et reprend au dernier checkpoint (arret apres 10 crashs ou si crash < 30s)
python recover_train.py --stage 3 --num_envs 1024
```

---

## 4. Deploiement Docker

Un `Dockerfile` et un `docker-compose.yml` complets avec support GPU NVIDIA sont prets a l'emploi :

```bash
# 1. Construction de l'image
docker compose build

# 2. Entrainement automatique complet avec GPU
docker compose run --rm go2-rl python train_all.py

# 3. Suivi en temps reel avec TensorBoard
docker compose up -d tensorboard
# Accessible sur http://localhost:6006

# 4. Visualisation 3D Web (Viser)
docker compose run --rm -p 8080:8080 go2-rl python play_simple.py --viewer viser
# Accessible sur http://localhost:8080

# 5. Visualisation fenetre native (X11)
xhost +local:root
docker compose run --rm -e DISPLAY=:1 go2-rl python play_simple.py
```

---

## 5. Documentation complete

Pour la liste exhaustive des options, drapeaux et astuces, consultez :
- [CHEAT_SHEET.md](CHEAT_SHEET.md) : Guide rapide de toutes les commandes.
- [USER_MANUAL.md](USER_MANUAL.md) : Manuel utilisateur complet et architecture detaillee.
