# ✈️ Prédiction des retards de vols dus aux conditions météorologiques

Ce projet de mémoire explore la prédiction des retards de vols, en mettant l'accent sur les retards causés par les conditions météorologiques. Il s'inscrit dans le contexte croissant du trafic aérien et de ses impacts économiques et logistiques.

---

## 🧠 Objectifs

- Distinguer les retards **dus à la météo** des **autres types de retards**.
- Comparer des modèles d'apprentissage profond pour prédire ces retards.
- Évaluer un modèle innovant : les **Liquid Neural Networks (LNN)**.
- Améliorer la performance des modèles malgré un **déséquilibre des classes**.

---

## 🗃️ Données utilisées

- **Données de vols** : Bureau of Transportation Statistics (États-Unis)  
- **Données météo** : Weather Underground  

Les deux sources ont été croisées pour créer un jeu de données riche et cohérent.

---

## 🛠️ Méthodologie

- **Prétraitement des données** : nettoyage, fusion, gestion des valeurs manquantes.
- **Déséquilibre des classes** : utilisation de **SMOTE** pour la suréchantillonnage.
- **Sélection des variables** : basée sur le score d’information mutuelle et la corrélation de Pearson.
- **Optimisation** : recherche par grille (`GridSearchCV`) + ajustements manuels.

---

## 🤖 Modèles évalués

- 🧪 **Liquid Neural Network (LNN)** – première application dans ce domaine  
- 🔁 **Long Short-Term Memory (LSTM)**  
- 🔢 **Multilayer Perceptron (MLP)**  

---

## 📊 Évaluation

Les modèles ont été comparés sur les métriques suivantes :
- Précision, rappel, F1-score, accuracy
- Matrice de confusion

📈 **Résultats clés** :
- LNN > LSTM/MLP pour la prédiction des vols à l’heure  
- Performance moyenne pour les retards **non liés à la météo**  
- Difficultés pour le LNN à bien prédire les retards **météo**, mais résultats prometteurs

---

## 📝 Conclusion

Le modèle **Liquid Neural Network** présente un fort potentiel malgré ses limitations (temps d'entraînement long, performance météo à améliorer). Il ouvre des perspectives intéressantes pour la **prédiction des retards de vols** en contexte réel.

---

## 📚 Technologies utilisées
- Python, Pandas, NumPy, Scikit-learn

- TensorFlow, PyTorch

- SMOTE (imbalanced-learn)

- Matplotlib, Seaborn

- Jupyter Notebook



---

## 📂 Structure du projet

```bash
📁 data/               # Jeux de données bruts et transformés
📁 notebooks/          # Notebooks d'analyse, de prétraitement, de modélisation
📁 models/             # Entraînements, hyperparamètres, modèles sauvegardés
📁 reports/            # Résultats, figures, matrice de confusion, métriques
📄 main.py             # Script principal d'entraînement / évaluation
📄 requirements.txt    # Dépendances Python

