## Readout : entraînement vs décision, avec une HPO par readout

Précision de la phrase suivante sur les chants de test, en %. Chaque readout a **sa propre HPO** : le réservoir est optimisé pour lui (400 essais, validation croisée sur les chants d'entraînement), puis évalué sur 10 graines. Les tableaux portent sur random_ei, chaque case donnant **500 / 1000 neurones**. Les lignes indiquent comment le readout est **entraîné**, les colonnes comment il **décide** au test. En gras : la diagonale (même mode des deux côtés). Surligné : le meilleur score.

### Sans fuite

| entraînement ↓ · décision → | last | mean | tail_mean 5 | tail_mean 10 |
|:--|--:|--:|--:|--:|
| last | **79,4 / 81,8** | 57,6 / 62,6 | 68,9 / 75,2 | 65,0 / 68,0 |
| mean | 34,2 / 35,6 | **81,4 / 82,9** | 65,5 / 64,7 | 77,8 / 78,2 |
| all | 57,7 / 65,7 | 68,1 / 72,5 | 64,3 / 70,0 | 67,2 / 71,5 |
| tail 5 | 75,1 / 78,5 | 64,1 / 70,2 | **79,0 / 81,0** | – |
| tail 10 | 72,1 / 76,7 | 72,0 / 75,8 | – | **77,8 / 79,1** |
| tail_mean 5 | 49,1 / 53,5 | 69,7 / 73,8 | **81,0 / 82,9** | – |
| tail_mean 10 | 38,6 / 45,2 | 75,7 / 78,7 | – | <mark>**82,2 / 83,7**</mark> |

### Avec fuite

| entraînement ↓ · décision → | last | mean | tail_mean 5 | tail_mean 10 |
|:--|--:|--:|--:|--:|
| last | <mark>**84,8 / 86,0**</mark> | 67,8 / 68,9 | 82,7 / 83,7 | 80,2 / 81,3 |
| mean | 71,1 / 72,2 | **81,8 / 83,0** | 74,4 / 75,4 | 80,3 / 80,9 |
| all | 72,1 / 74,1 | 77,6 / 78,6 | 74,9 / 76,7 | 76,7 / 78,1 |
| tail 5 | 82,9 / 84,2 | 72,4 / 75,0 | **82,6 / 84,1** | – |
| tail 10 | 82,0 / 83,5 | 75,4 / 77,1 | – | **80,9 / 82,3** |
| tail_mean 5 | 83,4 / 84,4 | 71,0 / 74,2 | **84,0 / 85,1** | – |
| tail_mean 10 | 82,5 / 83,3 | 76,1 / 79,1 | – | **83,9 / 84,8** |

### Sur tous les réservoirs

Écart moyen avec last → last sur 12 réservoirs (mean_hag, var_hag, random_ee et random_ei à 500 et 1000 neurones, plus les 4 hybrides à 1000), et nombre de réservoirs où le readout fait mieux :

| readout | sans fuite | avec fuite |
|:--|--:|--:|
| last → last | 81,7 % | 85,4 % |
| tail_mean 10 → tail_mean 10 | **+1,8** (12/12) | −0,9 (0/12) |
| tail_mean 5 → tail_mean 5 | +1,0 (11/12) | −0,8 (0/12) |
| mean → mean | +0,3 (4/12) | −3,1 (0/12) |
| tail 5 → tail_mean 5 | −0,7 (2/12) | −2,1 (0/12) |
| all → mean | −8,6 (0/12) | −7,0 (0/12) |

### À retenir

- **Décider comme on a entraîné.** Même avec un réservoir optimisé pour lui, un readout croisé perd de 2 à 40 points (mean → last, sans fuite). Ce n'était donc pas un artefact des réservoirs optimisés pour last → last.
- **Sans fuite, moyenner la fin de la phrase aide** : tail_mean 10 → tail_mean 10 bat last → last sur les 12 réservoirs (+1,8 point en moyenne, jusqu'à +5,0 sur random_ee à 500 neurones). L'état seul porte trop peu de mémoire.
- **Avec fuite, last → last reste le meilleur**, sur les 12 réservoirs. Le réservoir intègre déjà le passé, et moyenner n'apporte plus rien (−0,8 point au mieux). Meilleur score : 86,0 % (random_ei et var_hag+random_ei, 1000 neurones).
- **all est toujours mauvais** (−7 à −17 points) : entraîner sur tous les états de la phrase noie sa fin, là où se décide la suite.

> Les chiffres de last → last viennent de l'évaluation précédente, faite avant les petites corrections d'annotations du 3 novembre (7 lignes). Aucune de ces évaluations n'inclut les 41 réétiquetages du 7 novembre.

[details="Légende des modes"]
- **last** : le dernier état de la phrase
- **mean** : la moyenne des états de toute la phrase
- **all** : tous les états de la phrase, chacun une ligne d'entraînement
- **tail n** : les n derniers états, chacun une ligne d'entraînement
- **tail_mean n** : la moyenne des n derniers états, une seule ligne pour l'entraînement, ou la moyenne des sorties sur ces n états pour la décision
- **–** : combinaison non testée (la même longueur de queue sert à l'entraînement et à la décision)
[/details]
