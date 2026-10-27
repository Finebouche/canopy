# Prédiction de la phrase suivante : effet des deltas et du taux de fuite

Six campagnes (octobre 2026), code du commit `fbfa8a0` de la branche `canary` : variable `HAG_CANARY_DELTAS=0|1|2`, retirée ensuite. Depuis, la base de la prédiction de la phrase suivante est faite des MFCC seuls, avec et sans fuite. Ses études et ses résultats sont des copies des campagnes `nodeltas` ci-dessous, sous les noms `next_phrase_calls_leak_*` et `next_phrase_calls_*`, dans `outputs/canary/`.

| Entrée | Avec fuite | Sans fuite |
|---|---|---|
| MFCC seuls (13) | `next_phrase_calls_nodeltas_leak_*` | `next_phrase_calls_nodeltas_*` |
| MFCC + deltas d'ordre 1 (26) | `next_phrase_calls_delta1_leak_*` | `next_phrase_calls_delta1_*` |
| MFCC + deltas d'ordres 1 et 2 (39) | `next_phrase_calls_leak_*` | `next_phrase_calls_*` |

Fichiers :
- `*_hpo.csv` : résultats de test par condition et par graine ;
- `*_hpo_comparisons.csv` : comparaisons appariées ;
- `studies/` : les 72 études Optuna (non suivies par git) ;
- `*_hpo_parts/` : sur PlaFRIM seulement.

## Protocole
- **Tâche** : à la fin de chaque phrase, prédire l'étiquette de la phrase suivante (END après la dernière, START avant la première).
- **Phrases** : étiquettes répétées fusionnées, cris (`call`) inclus comme phrases, SIL et TRASH ignorés. 29 classes (phrases, `call`, END).
- **Découpage** : 367 chants d'entraînement / 92 chants de test (GroupShuffleSplit par chant). Transitions entre phrases comparables entre les deux ensembles (TVD moyenne 0.048, contre 0.049 ± 0.006 pour 1000 découpages aléatoires).
- **Entrée** : MFCC 44.1 kHz (fenêtre 23 ms, pas 11.6 ms, 500–8000 Hz), sans plancher `top_db`, mis à l'échelle sur l'entraînement.
- **Causalité** :
  - deltas calculés vers l'arrière (delta centré décalé de 4 trames) ;
  - point de prédiction 11.6 ms avant la fin annotée de la phrase (demi-fenêtre) ;
  - vérifié : aucune feature jusqu'au point ne dépend de l'audio après la fin de la phrase.
- **Modèle** :
  - réservoir sur tout le chant, sans remise à zéro ;
  - readout ridge linéaire entraîné aux points de prédiction ;
  - ridge choisi par validation croisée à 3 plis par chant.
- **HPO** : 400 essais par étude (Optuna TPE), objectif = exactitude hors pli. Avec ou sans taux de fuite cherché (sans : `leaky_rate` = 1).
- **Modèles** :
  - purs : mean_hag, var_hag, random_ee (ESN), random_ei (ESN excitateur/inhibiteur), à 500 et 1000 neurones ;
  - hybrides HAG + ESN à 1000 neurones : parties optimisées seules à 500 neurones (« parties »), puis optimisation jointe (« joint ») ;
  - contrôles : `_proj` (HAG sans apprentissage), readout sur les MFCC seuls.
- **Évaluation** : 92 chants de test, 10 graines, comparaisons appariées par graine (Wilcoxon).
- **Référence** : n-gramme adaptatif sur les vraies étiquettes = **77.1 %**.

## Exactitude en test, 1000 neurones (%, moyenne sur 10 graines)

| Condition | MFCC | MFCC, fuite | + Δ1 | + Δ1, fuite | + Δ1 + Δ2 | + Δ1 + Δ2, fuite |
|---|---|---|---|---|---|---|
| mean_hag+random_ei (joint) | 83.1 | 86.0 | 83.5 | **86.2** | 83.1 | 85.6 |
| var_hag+random_ei (joint) | 82.9 | 86.0 | 82.6 | 86.1 | 82.7 | 85.9 |
| var_hag+random_ee (joint) | 81.7 | 85.9 | 83.2 | 85.9 | 82.8 | 85.9 |
| random_ei | 81.6 | 85.6 | 81.4 | 85.9 | 81.4 | 85.9 |
| mean_hag | 83.0 | 85.3 | 83.2 | 85.6 | 83.2 | 85.8 |
| var_hag | 84.4 | 85.4 | 82.4 | 85.8 | 82.7 | 85.6 |
| random_ee | 79.0 | 85.6 | 81.4 | 85.7 | 81.4 | 85.7 |
| `_proj` | ~40 | 65–71 | 63–65 | 77–79 | 70–71 | 78–80 |
| MFCC seuls, sans réservoir | 35.3 | 35.3 | 56.1 | 56.1 | 64.0 | 64.0 |

Écart-type entre graines ≤ 1.1 point.

Avec fuite, les meilleurs modèles ont un F1 macro de ~81.5–81.8 % et un rappel de END de 11–16 % (2–9 % sans fuite).

## Exactitude en test, 500 neurones (%)

| Condition | MFCC | MFCC, fuite | + Δ1 | + Δ1, fuite | + Δ1 + Δ2 | + Δ1 + Δ2, fuite |
|---|---|---|---|---|---|---|
| random_ei | 78.9 | 84.6 | 79.2 | **84.8** | 79.4 | 84.5 |
| var_hag | 81.5 | 84.6 | 81.3 | 84.4 | 81.2 | 84.1 |
| mean_hag | 82.1 | 84.5 | 82.1 | 84.5 | 81.7 | 84.7 |
| random_ee | 76.9 | 84.3 | 79.4 | 84.4 | 79.4 | 84.5 |

Passer de 500 à 1000 neurones (avec fuite) rapporte +0.75 à +1.5 point.

## Conclusions

1. **Prédire au plus tard rapporte.** Le meilleur modèle atteint ~86 %, contre 85.7 % (deltas centrés) et 84.1 % (sans deltas) quand on prédisait 58 ms avant la fin.
2. **Le taux de fuite est indispensable** : +3.3 à +3.6 points en moyenne, significatif dans toutes les conditions (159–160 graines gagnantes sur 160).
3. **Avec fuite, les deltas ne servent plus** :
   - + Δ1 contre MFCC seuls : +0.14 point en moyenne (−0.24 à +0.61) ;
   - + Δ2 contre + Δ1 : −0.04 point (−0.60 à +0.51) ;
   - les deltas restent utiles sans mémoire : readout sur les MFCC seuls 35 → 56 → 64 %, `_proj` ~40 → ~70 % sans fuite ;
   - interprétation : au point le plus tard, le réservoir à fuite calcule lui-même la dynamique. Les deltas centrés compensaient surtout les 46 dernières ms non entendues.
4. **HAG contre ESN (random_ee)** :
   - avec fuite : HAG seul à égalité (±0.3 point) ; hybrides joints légèrement meilleurs (+0.2 à +0.5 point, souvent p < 0.05) ;
   - sans fuite : HAG nettement meilleur (+1 à +5 points). Écart maximal sur les MFCC seuls : mean_hag +4.0 à 1000 neurones, +5.2 à 500.
5. **END reste mal prédit** (rappel ≤ 16 %) : la phrase courante n'indique presque jamais la fin du chant (P(END | phrase) ≤ 0.24). La pondération des classes ne change pas ce résultat.
6. **Conséquence retenue** : point au plus tard, taux de fuite, MFCC seuls (13 entrées). C'est la nouvelle base. Meilleur résultat : hybride HAG + ESN joint à 1000 neurones, ~86 %, soit +9 points sur le n-gramme adaptatif.
