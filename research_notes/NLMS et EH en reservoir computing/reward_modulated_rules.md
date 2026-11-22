# Règles à modulation par la récompense et à perturbation (hors EH de Hoerzer 2014) pour entraîner des readouts de réservoirs / RNN

Légende de vérification :
- **[lu]** : vérifié dans le texte intégral (Europe PMC / HTML arXiv lu et cherché par mots-clés).
- **[résumé]** : vérifié seulement dans le résumé, une page de métadonnées ou un résumé de moteur de recherche.
- **[non vérifié]** : souvenir ou affirmation de seconde main, à confirmer.

Les PDF compressés ou scannés (Werfel 2003, Züge 2023, Widrow & Lehr 1990, Pyle & Rosenbaum en PDF) n'ont pas pu être lus directement : les points qui en viennent sont marqués en conséquence.

## Q1. Une règle à perturbation ou modulée par la récompense normalise-t-elle son pas par la norme de l'entrée ou de la perturbation (comme NLMS) ? Résultats théoriques sur le pas optimal

### Takeaway
Aucune des règles classiques (REINFORCE, perturbation de nœuds de Fiete & Seung, EH de Legenstein 2010, règle de Miconi 2017, R-STDP) n'utilise un pas η/(‖x‖² + c). L'exemple le plus proche trouvé est **Burms, Caluwaerts & Dambre 2015**, qui préconditionnent la règle RMH par (XᵀX + λI)⁻¹ calculé sur l'essai. Pour un seul pas de temps, cela revient exactement à un pas ε-NLMS (c = λ), et le taux d'apprentissage utilisable passe de 0,005 à 0,5. Deux autres familles normalisent autrement : Fernández et al. 2024 (ANP) divisent par la norme de la **perturbation d'activité** ‖δa‖², et Hiratani et al. 2022 normalisent les **poids**. En théorie (Werfel et al. 2005), le pas maximal des méthodes stochastiques varie comme l'inverse de la dimension du bruit injecté. Pour une sortie linéaire, la mise à l'échelle par ‖x‖² est de la même nature que pour LMS (inférence ci-dessous).

### Cited Findings

**Théorie du pas**
- [résumé] Werfel, Xie & Seung (NIPS 2003, pp. 1197–1204 ; version longue Neural Computation 17(12):2699–2718, 2005) comparent la descente de gradient directe, la perturbation de nœuds (NP) et la perturbation de poids (WP) sur un perceptron linéaire. Avec le pas réglé pour la vitesse maximale, NP est plus lente que le gradient d'un facteur égal au **nombre de sorties M**, et WP l'est encore plus d'un facteur égal au **nombre d'entrées N**. De plus, « the maximum learning rate for the stochastic methods scales inversely with the first power of the dimensionality of the noise injected ». Avec un pas assez petit, les trois méthodes ont les mêmes courbes d'apprentissage. Les auteurs précisent que ces conclusions dépendent de la distribution des entrées et de la cible. — [NIPS PDF](https://proceedings.nips.cc/paper_files/paper/2003/file/f8b932c70d0b2e6bf071729a4fa68dfc-Paper.pdf) ; [Princeton record](https://collaborate.princeton.edu/en/publications/learning-curves-for-stochastic-gradient-descent-in-linear-feedfor/)
- [résumé] Dans le résumé, aucune normalisation par ‖x‖² n'apparaît : le pas est un scalaire fixe optimisé. Les équations n'ont pas pu être lues (PDF compressé). — [NIPS PDF](https://proceedings.nips.cc/paper_files/paper/2003/file/f8b932c70d0b2e6bf071729a4fa68dfc-Paper.pdf)
- [résumé] Züge, Klos & Memmesheimer, *Weight versus node perturbation learning in temporally extended tasks: weight perturbation often performs similarly or better*, Physical Review X 13(2):021006 (2023) ; préprint bioRxiv 2021.10.04.463055. L'argument classique en faveur de NP (dimension de perturbation plus faible) ne tient plus quand la tâche s'étend dans le temps ou est apprise par lots, car cela « increases the perturbation dimension of NP but not WP ». WP fait aussi bien ou mieux quand la dynamique utilise peu de dimensions (beaucoup de solutions). Des perturbations corrélées dans le temps améliorent NP quand les entrées sont corrélées de façon semblable. Le texte intégral (règles et pas) n'a pas pu être lu. — [PRX](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.021006) ; [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.10.04.463055.full.pdf)
- [résumé] Hiratani, Mehta, Lillicrap & Latham, *On the Stability and Scalability of Node Perturbation Learning*, NeurIPS 2022. En théorie de champ moyen sur des réseaux linéaires à une couche cachée, le temps d'apprentissage dépend peu de la taille de la couche cachée. En revanche, NP reste **instable** si l'élève et le professeur diffèrent : la diffusion des poids les fait croître sans limite. « This instability can be suppressed by weight normalization, at the cost of bias in the learning rule. » Il s'agit d'une normalisation des **poids**, pas du pas par la norme de l'entrée. — [NeurIPS poster](https://neurips.cc/virtual/2022/poster/53156) ; [UCL PDF](https://discovery.ucl.ac.uk/10166318/1/10935_on_the_stability_and_scalabili.pdf)
- [résumé] Katahira et al. (*Optimal node perturbation in linear perceptrons with uncertain eligibility trace*) proposent un calendrier adaptatif de la variance de perturbation et du pas pour équilibrer robustesse au bruit et erreur résiduelle. Seul un résumé de moteur de recherche a été vu. — [Bohrium](https://www.bohrium.com/paper-details/optimal-node-perturbation-in-linear-perceptrons-with-uncertain-eligibility-trace/811820228747984898-2463)

**Normalisations trouvées dans des règles à perturbation ou RMH**
- [lu] **Burms, Caluwaerts & Dambre 2015**, *Reward-Modulated Hebbian Plasticity as Leverage for Partially Embodied Control in Compliant Robotics*, Frontiers in Neurorobotics 9:9 (doi 10.3389/fnbot.2015.00009), équipe Reservoir de Gand.
  - Règle de base, par essai : ΔW = α (r − r̄) Zᵀ X, où Z est le bruit d'exploration, X les états présynaptiques et r̄ la récompense prédite (moyenne des 50 dernières récompenses pour chaque entrée).
  - Variante « décorrélée » : **ΔW = α (r − r̄) Zᵀ X (XᵀX + λI)⁻¹**, avec λ = 1, plus une variante où un Heaviside H(r − r̄) ne garde que les mises à jour meilleures que prévu. Les auteurs : « This learning rule is similar to ridge regression ».
  - Pas : α = 0,005 pour la règle de base, contre α = 0,5 pour la règle décorrélée.
  - La décorrélation aide beaucoup quand le bruit d'exploration est corrélé (figure 12).
  - Tâches : réseaux tanh initialisés comme des réservoirs (XOR 2 bits, décodeur 3 bits, tâche continue), puis robot tenségrité **simulé** utilisé comme réservoir physique pour écrire des caractères. Bonne qualité après 1000 à 1500 essais.
  - Comparaisons : une variante où le bruit est estimé (« similar to the EH rule » de Legenstein 2010) et CMA-ES. CMA-ES est « nowhere near converging » quand RMH a trouvé une bonne solution. Pas de comparaison avec ridge, RLS ou FORCE.
  - L'annexe A décrit une variante stabilisée de type Oja, qui normalise la norme L2 de chaque vecteur de poids.
  — [Europe PMC PMC4538293](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4538293/)
- [lu, via le HTML arXiv] García Fernández, Keemink & van Gerven, *Gradient-Free Training of Recurrent Neural Networks using Random Perturbations* (arXiv 2405.08967, v3 octobre 2024).
  - La perturbation de nœuds « basée sur l'activité » (ANP) **divise par la norme au carré de la différence de pré-activation ‖δa_t‖²** et multiplie par le nombre de neurones N. Un pas η séparé est ensuite réglé par tâche.
  - Pas de normalisation par la norme de l'entrée.
  - Décorrélation des unités cachées par une règle non supervisée (variante DANP).
  - Tous les poids sont entraînés (entrée, récurrents et sortie) : il ne s'agit pas d'un réservoir.
  - Tâches : Mackey-Glass, copying memory, météo. ANP « closely approach[es] » BPTT+Adam, tandis que NP et WP restent « significantly inferior ».
  — [arXiv HTML v3](https://arxiv.org/html/2405.08967v3)
- [résumé] Dalm, van Gerven & Ahmad (arXiv 2310.00965, v8 juin 2026) relient NP à la **dérivée directionnelle** (perturbation ramenée à un vecteur unitaire, selon un résumé de moteur de recherche) et ajoutent une « input decorrelation at every layer ». Ils notent que NP standard « is highly data inefficient and can be unstable ». — [arXiv](https://arxiv.org/abs/2310.00965)
- [lu] Miconi 2017 (eLife 6:e20899) : pas fixe η (0,1, puis 0,03, puis 0,003 quand l'instant du stimulus varie), avec un **écrêtage** de chaque modification de poids par essai à |ΔJ| ≤ 10⁻⁴ (« roughly 10% of all potential weight modifications exceed this value and are clipped »). Il s'agit d'un écrêtage, pas d'une normalisation NLMS. — [Europe PMC PMC5398889](https://elifesciences.org/articles/20899)
- [lu, via Miconi 2017] Perturbation de nœuds (Fiete & Seung 2006, 2007) telle que résumée par Miconi : Δw_ij = η · e_ij · (R − R₀), avec e_ij l'accumulation du produit entrée × perturbation et R₀ la récompense attendue sans perturbation. Le pas η est fixe et il n'y a pas de normalisation. Miconi la présente comme une variante de REINFORCE (Williams 1992). — [Miconi 2017, eLife](https://elifesciences.org/articles/20899)
- [lu, via Miconi 2017] Règle E-H de Legenstein et al. 2010 : Δw_ij(t) = x_j(t) (y_i(t) − ȳ_i(t)) (R(t) − R̄(t)), sans normalisation. Hoerzer 2014 remplace R − R̄ par M(t) ∈ {0, 1}. — [Miconi 2017](https://elifesciences.org/articles/20899)
- [lu, via Pyle & Rosenbaum] Dans la réimplémentation de RMHL (Hoerzer) par Pyle & Rosenbaum, τ_w dW/dt = Φ(ê) ẑ rᵀ, avec τ_w = 0,02 et des fonctions Φ, Ψ en puissances 1/4 ou 1/5 de l'erreur. Aucune normalisation par la norme de l'entrée ou de la perturbation n'est décrite. — [ar5iv 1803.03304](https://ar5iv.labs.arxiv.org/html/1803.03304)
- [lu] Pogodin et al. 2019 : ΔW_ro = η(t) M(t) (z − z̄) rᵀ, avec η(t) = η₀/(1 + t/τ) et η₀ = 5·10⁻⁴. La décroissance est un calendrier temporel, sans normalisation par la norme. — [ar5iv 1910.10559](https://ar5iv.labs.arxiv.org/html/1910.10559)
- [résumé ; OCR peu fiable] Widrow & Lehr, *30 years of adaptive neural networks*, Proc. IEEE 78(9), 1990. Le même article présente α-LMS, la forme normalisée W_{k+1} = W_k + α ε_k X_k/|X_k|², et MRIII (Andes, Widrow, Lehr & Wan, IJCNN 1990), qui estime le gradient par perturbation Δs d'une Adaline et mesure la variation de l'erreur quadratique. Je n'ai **pas pu vérifier** si la mise à jour MRIII publiée est normalisée par |X_k|² ou utilise un μ fixe (PDF scanné illisible). — [Stanford PDF](https://www-isl.stanford.edu/~widrow/papers/j199030years.pdf) ; [diapos](https://pdfs.semanticscholar.org/bdc3/0f329df90cc887b85bb64febc422c5544b66.pdf)

### Inferences
- Pour un seul pas de temps (X = xᵀ, une ligne), Sherman-Morrison donne (xxᵀ + λI)⁻¹ x = x/(λ + ‖x‖²). La règle « décorrélée » de Burms et al. devient alors α (r − r̄) z xᵀ/(λ + ‖x‖²) : **une perturbation de nœuds à pas ε-NLMS**, avec c = λ. Sur un essai de T pas, c'est un préconditionnement de type projection affine ou ridge par essai. Les auteurs ne font pas ce rapprochement : ils parlent de décorrélation et d'analogie avec ridge. Le gain de pas d'un facteur 100 (0,005 → 0,5) va dans le sens de l'intuition NLMS.
- Pour un readout linéaire y = Wx + ξ avec un coût quadratique, l'espérance de la mise à jour NP vaut (à σ² près) le gradient LMS −(y − d) xᵀ. La condition de stabilité en moyenne porte donc sur η‖x‖², comme pour LMS. La variance ajoutée par l'estimation stochastique est proportionnelle à la dimension du bruit (Werfel). Si les entrées de Werfel sont isotropes, avec ‖x‖² ≈ N, leur loi « pas maximal ∝ 1/dimension » englobe implicitement un facteur 1/‖x‖². Cette dernière hypothèse sur la distribution des entrées n'est **pas vérifiée**.
- Dans un réservoir, ‖x(t)‖² varie fortement selon les entrées (silences et syllabes, par exemple). Un pas fixe η, comme dans EH, Miconi ou Pogodin, est donc trop grand dans les régions de forte activité et trop petit ailleurs. C'est précisément le défaut que corrige NLMS. Aucune publication trouvée ne teste cette hypothèse pour EH.

### Gaps
- Je n'ai trouvé aucun article qui applique explicitement « NLMS » ou η/(‖x‖² + c) à EH ou à la perturbation de nœuds sous ce nom. Je ne peux pas affirmer qu'aucun n'existe : les recherches ont été limitées (environ 15 requêtes).
- Équations exactes de Werfel 2005 (pas optimal en fonction de N, M et de la distribution des entrées) et de Züge 2023 (choix de η et σ) : PDF non lisibles ici.
- Normalisation de MRIII dans Widrow & Lehr 1990 : non vérifiée.
- Texte intégral de Legenstein, Chase, Schwartz & Maass 2010 (J Neurosci 30(25):8400–8410) : non accessible (Europe PMC n'a pas le texte, PMC demande un captcha). La règle est connue seulement par Miconi 2017 et par le résumé.
- Lillicrap et al. (*Backpropagation and the brain*, 2020) sur la variance des estimateurs par perturbation : non consulté.

## Q2. Ces règles ont-elles été appliquées à des caractéristiques étendues (polynomiales, noyaux, features aléatoires) d'un réservoir ?

### Takeaway
Non, dans tout ce qui a été trouvé. Les readouts appris par RMH ou par perturbation lisent directement les taux ou états du réservoir (tanh(x) + bruit), parfois après une décorrélation linéaire (Burms 2015). Les seules non-linéarités présentes servent au signal de récompense ou à la trace d'éligibilité : la fonction cubique S de Miconi, Φ et Ψ en puissances fractionnaires chez Pyle & Rosenbaum. Elles ne sont pas des caractéristiques d'entrée.

### Cited Findings
- [lu] Pyle & Rosenbaum : les entrées du readout sont r = tanh(x) + αη (N = 1000, p = 0,1, λ = 1,5). Aucune expansion non linéaire au-delà des taux tanh. — [ar5iv 1803.03304](https://ar5iv.labs.arxiv.org/html/1803.03304)
- [lu] Pogodin et al. 2019 : readout linéaire z = W_ro r + bruit sur les 1000 taux du réservoir. Les unités de mémoire de travail ne sont pas lues directement : elles agissent seulement en pilotant le réservoir. — [ar5iv 1910.10559](https://ar5iv.labs.arxiv.org/html/1910.10559)
- [lu] Burms et al. 2015 : readouts et poids de retour linéaires sur les états du réseau ou les capteurs du robot. La seule transformation est la décorrélation (XᵀX + λI)⁻¹. — [PMC4538293](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4538293/)
- [lu] Miconi 2017 : la non-linéarité supralinéaire S(x) = x³, ou x|x|, s'applique aux **incréments de plasticité** (entrée × fluctuation de sortie), pas aux entrées. Ce sont les poids récurrents qui sont appris, pas un readout fixe. — [Miconi 2017](https://elifesciences.org/articles/20899)
- [lu] Legenstein, Pecevski & Maass 2008 (PLoS CB 4(10):e1000180) : un neurone LIF de readout appris par R-STDP sur un microcircuit de 540 neurones LIF (chiffres parlés). Les entrées sont les trains d'impulsions du circuit, sans expansion. — [PMC2543108](https://ebusca.uv.mx/Record/pubmed-2543108/Details)
- [résumé] Article Neurocomputing 2023 (*Spatiotemporal motor learning with reward-modulated Hebbian plasticity in modular reservoir computing*) : les oscillations du réservoir sont « linearly integrated as readout output » avec des poids appris par RMHL, auxquels s'ajoute une trace d'éligibilité pour la récompense retardée. Auteurs non vérifiés (page 403). — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231223008639)

### Inferences
- Appliquer EH, la perturbation de nœuds ou NLMS à des monômes d'ordre 2 des états (comme `ridge:2`, `nlms:2`, `neh:2` dans canopy) semble **non publié** dans cette littérature. Avec des monômes, la dimension et la dispersion de ‖x‖² augmentent fortement, ce qui rend la normalisation du pas encore plus pertinente (voir Q1).
- La décorrélation de Burms et al. (2015) et celle de Fernández et Dalm (2023-2024) montrent que la corrélation des entrées est un problème connu de ces règles. Or les monômes d'ordre 2 sont très corrélés entre eux.

### Gaps
- Pas de recherche dédiée sur « kernel » ou « random features » combinés à des règles RL en RC, au-delà des requêtes générales. Un article isolé a pu être manqué.

## Q3. Travaux récents (2018–2026) comparant ces règles à FORCE, RLS ou ridge en reservoir computing, et leurs conclusions

### Takeaway
Peu de comparaisons directes. Elles concluent toutes que la règle RMH ou à perturbation est **plus lente et moins fiable** que FORCE ou le supervisé. RMHL de type Hoerzer oscille sans converger (Pyle & Rosenbaum 2019). RMH n'atteint le niveau de FORCE qu'avec une aide comme un réseau de mémoire de travail (Pogodin 2019). Les approches physiques par perturbation (2025) se comparent à la rétropropagation, pas à un readout ridge. Aucun banc d'essai trouvé ne compare RMH ou perturbation à ridge ou RLS sur un même réservoir avec des HPO séparées.

### Cited Findings
- [lu] **Pyle & Rosenbaum** (arXiv 1803.03304, 2018 ; publié dans Neural Computation 2019 selon un résumé de moteur de recherche, non vérifié), *A model of reward-modulated motor learning with parallel cortical and basal ganglia pathways*.
  - Tâches : dessin d'un papillon, en contrôle direct puis avec un bras à 2 ou 3 articulations.
  - FORCE « was able to quickly find the correct solution » sur la tâche 1, mais ne s'applique pas aux tâches 2 et 3 car il exige des cibles connues.
  - RMHL (Hoerzer) « relied on rapid changes in W to mimic the correct output … without truly learning it », et son W « continually oscillated rather than converging ».
  - Leur modèle SUPERTREX (voies parallèles) fait mieux.
  — [ar5iv](https://ar5iv.labs.arxiv.org/html/1803.03304)
- [lu] **Pogodin, Corneil, Seeholzer, Heng & Gerstner 2019** (arXiv 1910.10559, préprint), *Working memory facilitates reward-modulated Hebbian learning in recurrent neural networks*.
  - Pour des cibles de 1 s, la RMH du readout avec mémoire de travail atteint le niveau de FORCE après environ 90 essais.
  - Pour 10 s, elle atteint une performance de 1 en 200 essais, et échoue sans attracteur. Pendant ce temps, « The FORCE rule achieves a performance of 1 in all trials ».
  - Avec mises à jour retardées en fin d'essai : corrélation > 0,9 en moins de 100 essais.
  — [ar5iv](https://ar5iv.labs.arxiv.org/html/1910.10559)
- [lu] **Burms et al. 2015** (hors fenêtre, mais réservoir physique simulé) : RMH décorrélée contre une variante à bruit estimé de type EH et contre CMA-ES. RMH bat largement CMA-ES. Pas de baseline supervisée. Les auteurs notent que « in physical reservoir computing, embodied or morphological computation has always been exploited using supervised learning techniques ». — [PMC4538293](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4538293/)
- [résumé] **Abbott, Elo & Bozhko 2025** (arXiv 2506.04523, soumis à IEEE TNNLS), *Perturbative Gradient Training*. Des perturbations aléatoires dans l'espace des paramètres, avec des passes avant seulement, servent à entraîner des réseaux contenant un réservoir physique (un anneau magnonique en auto-oscillation). Les auteurs annoncent des performances « comparable to that of standard backpropagation ». Pas de baseline readout linéaire ni de détails sur l'estimateur dans le résumé. — [arXiv](https://arxiv.org/abs/2506.04523)
- [lu] **García Fernández et al. 2024** (RNN entièrement entraînés, pas de RC) : ANP et DANP approchent BPTT, alors que NP et WP en restent loin. Pas de comparaison avec FORCE, RLS ou ridge. — [arXiv HTML](https://arxiv.org/html/2405.08967v3)
- [résumé] Neurocomputing 2023 (RMHL dans un réservoir modulaire) : avec un réservoir oscillant et une trace d'éligibilité, RMHL apprend le timing moteur et le dessin de motifs là où les versions antérieures échouaient. Comparaison seulement avec d'anciens modèles RMHL, d'après l'extrait. — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231223008639)
- [résumé] Krenzer & Bogdan 2025 (Frontiers in Computational Neuroscience), *Reinforced Liquid State Machine* : LSM approfondie, couche winner-takes-all et synapses R-STDP récompensées selon la prédiction du readout. Étude logicielle, pas sur puce. — [Frontiers](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1569374/full)
- [lu] Legenstein, Pecevski & Maass 2008 (PLoS CB) : le readout LIF appris par R-STDP n'atteint pas la reconnaissance de mots indépendante du locuteur obtenue avec un readout linéaire supervisé. « further work will be needed » pour savoir si un readout appris par renforcement peut égaler les perceptrons supervisés. — [PMC2543108](https://ebusca.uv.mx/Record/pubmed-2543108/Details)
- [lu, via Miconi] Miconi 2017 : selon lui, la RMH simple « does not work for strongly-connected recurrent networks ». EH exige une récompense continue en temps réel, ce qui « negates a central advantage of reinforcement learning ». Sa règle (trace supralinéaire) entraîne les poids récurrents avec des récompenses rares et retardées, sur des tâches cognitives et sur le contrôle d'un bras. — [Miconi 2017](https://elifesciences.org/articles/20899)
- [résumé] Fiete & Seung 2006 (PRL 97:048104) : gradient estimé en perturbant dynamiquement les conductances des N neurones (généralisation de NP), interprétable comme une règle synaptique avec des synapses « empiric ». Fiete, Fee & Seung 2007 (pp. 2038–2057, J Neurophysiol selon mes souvenirs, non confirmé) : modèle d'apprentissage du chant (HVC, RA, LMAN) par apprentissage par renforcement fondé sur le gradient. Article **pertinent pour les canaris**. — [arXiv q-bio/0601028](https://arxiv.org/abs/q-bio/0601028) ; [PubMed](https://pubmed.ncbi.nlm.nih.gov/16907616/)
- [résumé] Une recherche sur les puces neuromorphiques n'a trouvé aucun article 2021-2024 combinant un readout R-STDP et un LSM sur puce. Les travaux sur puce trouvés (Liu, Yenamachintala & Li) utilisent une STDP supervisée modulée par le calcium. — [JETC 2018 PDF](https://web.ece.ucsb.edu/Faculty/li/Paper4-ACM-JETC-2018%20(1).pdf)

### Inferences
- La littérature ne fournit pas de comparaison **équitable** (au sens de canopy : HPO séparée par règle) entre EH, la perturbation de nœuds, NLMS et ridge sur un même réservoir. Les comparaisons existantes utilisent un seul réservoir réglé pour FORCE ou Hoerzer (Pyle & Rosenbaum, Pogodin). Elles ne permettent donc pas de classer les règles.
- Les échecs reportés de RMHL (oscillations sans convergence, sensibilité aux corrélations d'entrée) sont cohérents avec un problème de **pas mal adapté** à l'échelle des entrées. Ils viennent aussi de la variance de l'estimateur. Cela soutient l'intérêt d'un pas normalisé de type NLMS ou décorrélé à la Burms. C'est une hypothèse, non testée dans ces articles.

### Gaps
- Le texte intégral de Pyle & Rosenbaum dans Neural Computation 2019 n'a pas été vérifié (lu seulement sur arXiv), ni celui de l'article Neurocomputing 2023 (auteurs inconnus, page 403).
- Izhikevich 2007 (Cerebral Cortex 17, *Solving the distal reward problem through linkage of STDP and dopamine signaling*) : connu seulement par sa référence. Pas lu.
- Williams 1992 (REINFORCE) et Werfel 2005 : pas lus en texte intégral.
- Pas trouvé de réservoir photonique, électronique ou spintronique entraîné par une règle RMH ou NP **sur le readout** avec une comparaison chiffrée à ridge. Un préprint photonique de 2022 (arXiv 2202.12896) note que le RC matériel n'avait alors pas été appliqué à l'apprentissage par renforcement (résumé seulement).
