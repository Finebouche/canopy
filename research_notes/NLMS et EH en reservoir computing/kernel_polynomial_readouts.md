# Filtres adaptatifs à noyau et readouts polynomiaux (monômes) entraînés en ligne sur des états de réservoir

Périmètre : existe-t-il une publication d'une règle LMS/NLMS en ligne sur les monômes d'ordre 2 de l'état du réservoir (équivalent : KNLMS/KLMS normalisé à noyau polynomial en forme duale, noyau des monômes centrés) comme readout de réservoir ? Travaux les plus proches. Recherche faite le 2026-11-16 (environ 25 requêtes et lectures). Légende : **[lu]** = texte intégral lu (extrait PDF) ; **[résumé]** = résumé, métadonnées ou extraits de moteur seulement.

## Q1. Filtres adaptatifs à noyau (KLMS, KNLMS, KAPA, KRLS, QKLMS) appliqués sur des états de réservoir/ESN

### Takeaway
Un seul travail trouvé met explicitement un filtre adaptatif à noyau en ligne sur les états d'un ESN : ES-KRLS (Zhou et al. 2018), avec un **noyau gaussien**, KRLS et sparsification ALD. Deux autres lignées font de l'apprentissage à noyau sur des réservoirs (processus gaussiens en ligne d'écho, Soh & Demiris 2015 ; noyaux récursifs « ESN infinis », Hermans & Schrauwen 2012), mais aucun travail trouvé n'utilise KLMS/KNLMS, ni un noyau polynomial, sur des états de réservoir.

### Cited Findings
- **ES-KRLS** — Zhou H., Huang J., Lu F., Thiyagalingam J., Kirubarajan T., « Echo state kernel recursive least squares algorithm for machine condition prediction », *Mechanical Systems and Signal Processing* 111 (2018) 68–86, doi:10.1016/j.ymssp.2018.03.047 — [notice Liverpool](https://livrepository.liverpool.ac.uk/id/eprint/3019825) ; [manuscrit accepté, lu](https://livrepository.liverpool.ac.uk/3019825/1/ESN_JThiyagalingam.pdf). **[lu]**
  - Méthode : réservoir ESN fixe, puis le readout est « reformulated as a linear model in RKHS » et mis à jour à chaque pas par KRLS ; le readout devient un réseau RBF croissant, limité par une sparsification en ligne avec le critère ALD (dictionnaire d'« augmented inputs »). En ligne (« sequential learning approach »).
  - Noyau : **gaussien** pour ES-KRLS, KRLS et KLMS (section 5). Aucun noyau polynomial.
  - Tâches : Mackey-Glass (réservoir 300 unités, rayon spectral 0,99), laser Santa Fe (600 unités), puis pronostic de durée de vie restante (turbofan) avec une méthode bayésienne.
  - Résultats Mackey-Glass sans bruit (moyenne sur 50 réservoirs) : RMSE à 84 pas : ES-KRLS 1,86e-3, RLS-ESN 2,73e-2, SVESM 2,24e-2, KRLS seul (entrée fenêtrée) 9,55e-2, KLMS seul 2,85e-1, ELM 9,68e-2. Les hyperparamètres du réservoir sont réglés pour maximiser RLS-ESN (donc pas une HPO par méthode).
  - Revue de l'état de l'art dans ce papier : « the readout layer is traditionally constructed as linear adaptive filters, including the recursive least squares (RLS) and least mean square (LMS) » (renvoi à Jaeger 2002) ; le readout étendu à un filtre de Volterra (Boccato et al. 2012) est cité avec la remarque que son coût « expands exponentially with the number of hidden units ». KLMS n'y apparaît que comme référence comparée sur l'entrée brute, pas sur les états du réservoir.
- **OESGP / OIESGP** — Soh H., Demiris Y., « Spatio-Temporal Learning With the Online Finite and Infinite Echo-State Gaussian Processes », *IEEE TNNLS* 26(3) (2015) 522–536 : GP parcimonieux en ligne, à budget fixe, sur les états d'un ESN (ne garde que les états « informatifs » ou « nouveaux ») ; la variante infinie utilise un noyau récursif avec ARD, hyperparamètres adaptés par gradient naturel stochastique ; prédiction à un pas, identification de systèmes, apprentissage par démonstration en robotique (Nao, fauteuil ARTY) — [Spiral Imperial](https://spiral.imperial.ac.uk/handle/10044/1/18527) ; [dblp](https://dblp.uni-trier.de/rec/journals/tnn/SohD15.html). **[résumé]**
- **Echo state Gaussian process** — Chatzis S.P., Demiris Y., *IEEE TNN* 22 (2011) : readout bayésien (GP) sur états d'ESN, hors ligne — cité dans [ES-KRLS](https://livrepository.liverpool.ac.uk/3019825/1/ESN_JThiyagalingam.pdf) (réf. 28) et [notice](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_ieee_primary_5966349). **[résumé]**
- **Support vector echo-state machine (SVESM)** — Shi Z., Han M., *IEEE TNN* (2007) : readout SVM (coûts ε-insensible/Huber) sur états d'ESN, hors ligne ; comparé dans ES-KRLS — [ES-KRLS, réf. 22](https://livrepository.liverpool.ac.uk/3019825/1/ESN_JThiyagalingam.pdf). **[lu, via citation]**
- **Recurrent kernel machines** — Hermans M., Schrauwen B., « Recurrent kernel machines: computing with infinite echo state networks », *Neural Computation* 24 (2012), doi:10.1162/NECO_a_00200 : l'ESN de taille infinie définit un noyau récursif sur les séquences, utilisé dans des SVM récursives (hors ligne). Ici le noyau remplace le réservoir ; ce n'est pas un noyau appliqué à des états finis — [MaRDI](https://portal.mardi4nfdi.de/wiki/Publication:2885086). **[résumé]**
- Hors réservoir : Candido, Eisencraft, Silva (JCIS 34(1), 2019) utilisent KLMS/QKLMS à noyau gaussien pour l'égalisation de communications chaotiques, sans réservoir — [PDF](https://pdfs.semanticscholar.org/d37c/7007dddef71dd34badf3f8ab540043949149.pdf). **[lu]** (écarté, cité pour montrer qu'une requête « KLMS + chaos » ne mène pas à un readout de réservoir.)
- KNLMS — Richard C., Bermudez J.C.M., Honeine P., « Online prediction of time series data with kernels », *IEEE TSP* 57(3) (2009) 1058–1067 : critère de cohérence pour borner le dictionnaire, algorithme KAPA dont KNLMS est un cas particulier ; appliqué à des séries temporelles brutes (pas de réservoir) — [HAL](https://hal.archives-ouvertes.fr/hal-01965587). **[résumé]**
- Une revue RC observe que le « kernel trick » est en général non utilisé en RC, sauf via des noyaux temporels récursifs — [Lukoševičius et al. 2012](https://www.ai.rug.nl/minds/uploads/2508_Lukoseviciusetal12.pdf). **[résumé, extrait de moteur]**

### Inferences
- L'antériorité la plus proche pour « filtre adaptatif à noyau en ligne sur états de réservoir » est ES-KRLS (2018). Elle diffère de notre règle sur trois points : KRLS (second ordre, O(M²) par pas) et non NLMS ; noyau gaussien et non polynomial de degré 2 ; dictionnaire parcimonieux ALD. L'idée générique « ESN + KAF » est donc publiée ; la combinaison précise « NLMS + noyau polynomial / monômes d'ordre 2 » ne l'est pas, à notre connaissance.
- OESGP est aussi un apprentissage à noyau en ligne sur états d'ESN (filtre bayésien, parcimonieux), à mentionner comme antériorité de second rang.

### Gaps
- Je n'ai pas trouvé de papier KLMS, KNLMS, KAPA ou QKLMS appliqué aux états d'un ESN (requêtes « kernel least mean square » + « echo state network », « QKLMS reservoir »). Cela ne prouve pas l'absence (papiers de conférence chinois ou d'ingénierie mal indexés possibles). Une vérification dans Google Scholar (« KLMS » « echo state », « kernel adaptive filter » « reservoir computing ») et dans les citants d'ES-KRLS est recommandée.
- Références classiques KLMS (Liu, Pokharel & Príncipe, IEEE TSP 2008), KRLS (Engel, Mannor & Meir, IEEE TSP 2004), QKLMS (Chen et al. 2012) : non relues ici ; citées d'après ES-KRLS (réf. [6], [8]) et la littérature connue.
- Texte intégral de Soh & Demiris 2015 et Hermans & Schrauwen 2012 non lu.

## Q2. Readouts non linéaires / polynomiaux entraînés en ligne sur des réservoirs (quadratiques, NG-RC, Volterra)

### Takeaway
Les readouts quadratiques (états au carré, monômes, Volterra, NG-RC) sont nombreux, mais tous ceux vérifiés ici sont entraînés hors ligne (moindres carrés, ridge, PCA puis régression). L'apprentissage en ligne des readouts (LMS, RLS, FORCE) est classique mais concerne des readouts linéaires. Je n'ai pas trouvé de papier qui entraîne en ligne un readout quadratique ou à monômes sur un réservoir, ni un NG-RC entraîné par RLS ou LMS.

### Cited Findings
- **Readout de Volterra + PCA** — Boccato L., Lopes A., Attux R., Von Zuben F.J., « An extended echo state network using Volterra filtering and principal component analysis », *Neural Networks* (2012), doi:10.1016/j.neunet.2012.02.028 ; version IJCNN 2011. La PCA réduit le nombre de signaux envoyés au readout de Volterra, donc le nombre de coefficients ; tâches : égalisation de canal, séparation de sources — [Bohrium](https://www.bohrium.com/paper-details/an-extended-echo-state-network-using-volterra-filtering-and-principal-component-analysis/813330757303926784-2463) ; [researchr IJCNN 2011](https://researchr.org/publication/BoccatoLAZ11). **[résumé]** C'est l'antériorité la plus proche d'un « readout à monômes d'ordre 2 sur états d'ESN » (bien avant Ohkubo & Inubushi). Mode d'entraînement (en ligne ou hors ligne) non vérifié.
- Le même groupe a comparé un readout de Volterra et un readout ELM (égalisation, prédiction chaotique) : le readout non linéaire « can be decisive » — [extrait de recherche](https://ieeexplore.ieee.org/document/6252704/references). **[résumé]**
- **Bollt (2021)** : un VAR linéaire avec readout quadratique (produit de Hadamard) équivaut à un VAR non linéaire quadratique ; readout quadratique sur réservoir linéaire, ajusté par moindres carrés — [arXiv:2008.06530](https://arxiv.org/pdf/2008.06530). **[résumé]**
- **NG-RC** — Gauthier D.J., Bollt E., Griffith A., Barbosa W.A.S., *Nature Communications* 12 (2021) 5564 : NVAR, features = délais + monômes (quadratiques), seul le readout est appris par « regularized linear least-squares » (hors ligne) — [Nature Comms](https://www.nature.com/articles/s41467-021-25801-2) ; [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8455577). **[résumé]**
- **Adaptive NVAR** (arXiv:2507.08738, 2025) : remplace les monômes de NVAR par un petit MLP appris ; le papier note que le readout de NVAR reste calculé par moindres carrés — [arXiv](https://arxiv.org/html/2507.08738v2). **[résumé]** Pas d'entraînement en ligne du readout constaté.
- Readout en ligne linéaire : Jaeger (NIPS 2002, « Adaptive nonlinear system identification with echo state networks ») associe ESN et RLS — [NeurIPS](https://papers.nips.cc/paper_files/paper/2002/file/426f990b332ef8193a61cc90516c1245-Metadata.json). FORCE (RLS modifié) reste linéaire ; un papier 2026 l'utilise pour un contrôleur ESN de moteurs BLDC — [Prist et al. 2026](https://iris.univpm.it/retrieve/f61407d9-f1de-4d61-a5bc-ac40ff9891b1/Prist_Online-adaptation-Echo-State-Network_2026.pdf). **[résumé]** (Hors périmètre, cité comme base.)
- Une revue RC note que RLS est préféré à LMS en ligne pour sa robustesse à la dispersion des valeurs propres — [Survey arXiv:2307.15092](https://arxiv.org/pdf/2307.15092). **[résumé]** C'est pertinent pour nous : les monômes d'ordre 2 ont une matrice de corrélation encore plus mal conditionnée que les états, ce qui pénalise LMS/NLMS.
- Un papier IEEE intitulé « Online Algorithms for Polynomial Regression on Physical Reservoir Computers with Noisy Measurements » (LMS et RLS modifiés, garanties de convergence, états bruités) — [IEEE 9683633](https://ieeexplore.ieee.org/document/9683633/). **[résumé, extrait de moteur seulement]** La page n'a pas pu être lue ; auteurs, sens exact de « polynomial regression » (sur les états ? sur l'entrée ?) et monômes utilisés **non vérifiés**. C'est le candidat le plus sérieux d'antériorité « readout polynomial en ligne sur réservoir » ; à vérifier en priorité (CDC 2021 probable, groupe de H. Nurdin ? non confirmé).

### Inferences
- Le concept « features quadratiques de l'état + régression linéaire » est ancien en RC (Volterra/Boccato 2011-2012, états au carré de Pathak/Lu, Bollt 2021, NG-RC 2021, Ohkubo & Inubushi 2024). Il est toujours ajusté hors ligne dans les travaux vérifiés.
- Entraîner ces mêmes features en primal avec RLS/LMS est une extension directe (les monômes ne sont que des régresseurs de plus). Elle a pu être faite sans être présentée comme une contribution. La nouveauté défendable de notre variante porte donc moins sur « en ligne » que sur la forme duale (noyau polynomial, coût en O(N) par évaluation de noyau au lieu de O(N²) features) et sur le centrage des monômes.

### Gaps
- Pathak et al. 2018 (PRL, états d'indice impair élevés au carré) et Lu et al. 2017 (Chaos) : non relus ici. Le mode d'entraînement (ridge hors ligne) est connu mais non vérifié dans cette session.
- Le papier IEEE 9683633 n'a pas été lu (page vide via l'outil) : à lire.
- Pas de résultat pour « NG-RC » + « RLS » ou « LMS » en ligne. Un NG-RC entraîné par RLS existe peut-être dans la littérature contrôle et identification (2022-2026) sans être indexé sous ces mots-clés.

## Q3. Suites et citants d'Ohkubo & Inubushi 2024 : entraînement en ligne ou forme duale ?

### Takeaway
Le readout généralisé est ŷ = W r + rᵀ W^Q r, appris par ridge hors ligne. Le papier ne mentionne ni apprentissage en ligne, ni noyau. Parmi ses 21 citants recensés par Semantic Scholar (au 2026-11-16), aucun n'entraîne ce readout en ligne ni en forme duale/noyau, d'après titres et résumés.

### Cited Findings
- Ohkubo A. (arXiv : « Ookubo »), Inubushi M., « Reservoir computing with generalized readout based on generalized synchronization », *Scientific Reports* 14, 30918 (2024), doi:10.1038/s41598-024-81880-3 ; arXiv:2405.14885 — [arXiv PDF, lu](https://arxiv.org/pdf/2405.14885) ; [Sci. Rep.](https://www.nature.com/articles/s41598-024-81880-3). **[lu]**
  - Motivation : synchronisation généralisée, h(r) ≈ développement de Taylor ; le readout quadratique approche le hessien, W^Q_ijk ≃ ½ ∂²h_i(0)/∂r_j∂r_k (W^Q ∈ M_{L×N×N}, pas de symétrisation discutée). Variante cubique (C-ESN) mentionnée.
  - Apprentissage : « learning in our method is linear with respect to the weights », moindres carrés ; « we use the Ridge regression at the training phase ». Hors ligne.
  - Expériences : ESN tanh, N = 10 (balayage 5–40), Lorenz ; prédiction à court terme (τ = 0,2) et autonome en boucle fermée. Le Q-ESN reste précis environ 8 temps de Lyapunov contre environ 3 pour l'ESN linéaire, avec une robustesse accrue aux tirages du réservoir (chiffres d'après la lecture du HTML arXiv).
  - Pas de mention d'apprentissage en ligne, RLS, NLMS, noyau ni NG-RC ; Pathak et al. 2017 cité pour N = 300 avec readout linéaire.
- Citants (API Semantic Scholar, 21 papiers) — [S2 API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41598-024-81880-3/citations?fields=title,year,venue,authors,abstract&limit=100). **[résumé]** Les plus proches :
  - Singh P. et al., « Topologically informed echo state networks via Poincaré return maps… », *Neural Networks* (2026) : readout quadratique « once-trained » (hors ligne), jusqu'à 1,9× plus de temps de prédiction valide.
  - Singh P. et al., HypER (ECAI 2025) et « Swirl-Adaptive Hyperbolic Small-World Reservoir… » (Physica A 2026) : topologies de réservoir, pas de readout en ligne d'après les titres.
  - Pyragas V. et al., « Enhancing stability in next-generation reservoir computing by enforcing the echo-state property » (Physics Letters, 2026) : NG-RC, titre seul.
  - Kovalev A. et al., *Photonics* (2025) : transformation non linéaire des sorties de nœuds et suréchantillonnage en post-traitement (RC expérimental), environ 10× moins d'erreur sur Santa Fe.
  - Chen L. et al. (arXiv 2026) : combinaison de readouts pré-entraînés pour extrapoler à travers des bifurcations.
  - Hara M. (SIAM J. Appl. Dyn. Syst., 2026), Hart A.G. (*Chaos*, 2025), Bizyaeva A. (arXiv 2026, observateur KKL) : théorie.
  - Aucun titre ni résumé ne mentionne RLS, LMS, NLMS, noyau ou apprentissage en ligne du readout généralisé.
- Une revue (arXiv:2504.11757, 2025) le décrit comme un readout à combinaisons non linéaires (quadratiques, cubiques) des variables du réservoir — [arXiv](https://arxiv.org/html/2504.11757v1). **[résumé]**

### Inferences
- Le readout d'Ohkubo-Inubushi (linéaire + quadratique) correspond, en forme duale, à un noyau polynomial inhomogène de degré 2, k(r,r') = (c + rᵀr')², à une pondération près des features (facteurs √2 et c). Notre version centrée correspond à k(r,r') = ⟨φ(r) − μ, φ(r') − μ⟩. À notre connaissance, aucun citant ne fait ce rapprochement.
- Point de vigilance : Semantic Scholar n'indexe pas tout (thèses, conférences japonaises). Le résultat « aucun citant en ligne/dual » vaut au 2026-11-16 pour cette base.

### Gaps
- Citations Google Scholar non vérifiées (souvent plus nombreuses que celles de S2).
- Le chiffre exact (8 contre 3 temps de Lyapunov) vient du résumé du HTML arXiv par l'outil ; le PDF lu confirme la formule et le ridge, pas ces chiffres.

## Q4. Équivalent classique hors RC : Volterra NLMS d'ordre 2 et NLMS à noyau polynomial

### Takeaway
Le filtre de Volterra d'ordre 2 adapté par NLMS (forme primale, sur une ligne à retard) est standard en traitement du signal (annulation d'écho non linéaire). La littérature signale le problème de normalisation propre au noyau quadratique, qui converge lentement sous un pas commun. En forme duale, NLMS sur les features polynomiaux correspond à un KLMS normalisé à noyau polynomial. Le KNLMS de Richard et al. (2009) en est proche mais ajoute un dictionnaire par cohérence et une projection affine.

### Cited Findings
- Volterra d'ordre 2 + NLMS pour l'annulation d'écho acoustique non linéaire (minimisation d'un coût, pas NLMS dans (0, 2)) — [ATN 2013](https://users.utcluj.ro/~atn/papers/ATN_1_2013_1.pdf) ; structures découplées et NLMF — [ATN 2012](https://users.utcluj.ro/~atn/papers/ATN_2_2012_9.pdf). **[résumé]**
- Sous LMS, les coefficients quadratiques h2 bougent « in very small steps », d'où une normalisation séparée pour les noyaux d'ordre 1 et 2 — [Bilkent, NSIP 1999](https://www.ee.bilkent.edu.tr/~signal/Nsip99/papers/146.pdf). Pas variable selon l'excitation pour le noyau quadratique, comparé au Volterra-NLMS d'ordre 2 standard — [EUSIPCO 2012](https://new.eurasip.org/Proceedings/Eusipco/Eusipco2012/Conference/papers/1569588493.pdf). **[résumé]**
- Volterra LMS d'ordre 2 en manuel (bornes de pas liées à la matrice de corrélation de l'entrée) — [chapitre Springer](https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/978-0-387-68606-6_11.pdf). **[résumé]**
- KNLMS (Richard, Bermudez, Honeine 2009) : dictionnaire contrôlé par cohérence, KAPA, KNLMS cas particulier, séries temporelles — [HAL](https://hal.archives-ouvertes.fr/hal-01965587). **[résumé]**

### Inferences
- Notre règle (NLMS dual sur monômes d'ordre 2 centrés de l'état du réservoir) est l'analogue exact du « Volterra NLMS d'ordre 2 » où la ligne à retard est remplacée par l'état du réservoir, et du KLMS normalisé à noyau polynomial. L'avertissement de la littérature Volterra (normalisation séparée des termes linéaires et quadratiques) s'applique directement : avec un pas commun normalisé par ‖φ‖², les termes quadratiques dominent la norme.
- Les classiques KLMS (Liu, Pokharel & Príncipe 2008) prouvent la régularisation implicite de KLMS. Ils utilisent surtout le noyau gaussien ; le noyau polynomial y est mentionné comme cas possible (non vérifié ici).

### Gaps
- Mathews & Sicuranza, *Polynomial Signal Processing* (2000), référence pour Volterra LMS/NLMS/RLS : non consultée.
- L'équivalence formelle « NLMS primal sur features polynomiaux = KLMS normalisé dual à noyau polynomial » n'a pas été trouvée énoncée telle quelle dans une source lue. Elle découle de l'identité ⟨φ(x), φ(y)⟩ = k(x,y) (Liu, Príncipe & Haykin, *Kernel Adaptive Filtering*, 2010, présente un « normalized KLMS » ; non vérifié ici).
