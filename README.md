# Unitree Go2 Reinforcement Learning (MuJoCo Warp / mjlab)

Framework d'apprentissage par renforcement léger, optimisé et épuré pour le robot quadrupède **Unitree Go2**, entraîné sur des **terrains extérieurs complexes** (marches, terrains accidentés, murs, longs couloirs, trous).

Ce projet a été nettoyé pour ne conserver que les composants indispensables au Go2.

---

## ⚡ Démarrage Rapide

### 1. Entraîner le Go2
```bash
# Entraînement direct sur la Big Map complexe
python train_simple.py

# Ou avec le superviseur auto-recovery (relance automatique en cas de crash)
python recover_train.py
```

### 2. Jouer / Piloter le Go2
```bash
# Charge automatiquement le dernier checkpoint entraîné
python play_simple.py
```
> **Contrôles clavier (cliquez dans la fenêtre graphique pour focus) :**
> - **Flèche HAUT / W** : Avancer
> - **Flèche BAS / S** : Reculer
> - **Flèche GAUCHE / A** : Tourner à gauche
> - **Flèche DROITE / D** : Tourner à droite
> - **Espace / X** : Arrêt complet

---

## 🎓 Curriculum en 3 Étapes

Pour un entraînement progressif optimal :

1. **Étape 1 : Sol Plat**
   ```bash
   python train_simple.py --stage 1 --max_iterations 1500 --num_envs 1024
   ```
2. **Étape 2 : Escaliers et Trous**
   ```bash
   python train_simple.py --stage 2 --resume --max_iterations 3000 --num_envs 1024
   ```
3. **Étape 3 : Big Map Extérieure Complexe**
   ```bash
   python train_simple.py --stage 3 --resume --max_iterations 6000 --num_envs 1024
   ```

---

## 🛡️ Superviseur Anti-Crash (`recover_train.py`)

Gère les aléas matériels ou plantages GPU :
```bash
# Surveille le processus et reprend au dernier checkpoint (arrête après 10 crashs ou si crash immédiat)
python recover_train.py --stage 3 --num_envs 1024
```

---

## 🐳 Déploiement Docker

Un `Dockerfile` et un `docker-compose.yml` complets avec support GPU NVIDIA sont prêts à l'emploi :

```bash
# 1. Construction de l'image
docker compose build

# 2. Entraînement avec GPU
docker compose run --rm go2-rl python recover_train.py --stage 3

# 3. Visualisation 3D Web (Viser)
docker compose run --rm -p 8080:8080 go2-rl python play_simple.py --viewer viser
```
Ouvrez ensuite `http://localhost:8080` dans votre navigateur.

---

## 📖 Documentation complète

Pour la liste exhaustive des options, drapeaux et astuces, consultez le [CHEAT_SHEET.md](CHEAT_SHEET.md).
