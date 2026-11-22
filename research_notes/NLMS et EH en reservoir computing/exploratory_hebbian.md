# La règle « exploratory Hebbian » (EH) de Hoerzer, Legenstein & Maass (2014) et ses descendants

Portée : règle EH appliquée aux readouts de réservoirs, avec ses variantes, ses ré-implémentations et ses comparaisons. Les autres règles hebbiennes modulées par la récompense (node perturbation, Legenstein 2010 en dehors de son lien avec EH, Miconi 2017, Fiete & Seung) relèvent d'un autre chercheur ; elles ne sont citées ici que par leur lien avec EH.

Légende de vérification :
- **[lu]** : texte intégral lu, dans le PDF ou dans la version HTML de l'article.
- **[méthodes lues]** : section Méthodes lue, mais pas tout l'article.
- **[résumé]** : seulement le résumé, une page d'index ou un résumé fait par l'outil de recherche.

## Q1. Règle EH exacte de Hoerzer et al. 2014 : hyperparamètres et normalisations

### Takeaway
EH est un perceptron linéaire sans tanh, avec ou sans biais. Sa mise à jour est Δw_i(t) = η(t) · [z_i(t) − z̄_i(t)] · M(t) · r(t). Le bruit d'exploration est estimé par z − z̄, où z̄ est la sortie bruitée filtrée passe-bas. Le signal M(t) est **binaire** et **global** : il vaut 1 si P(t) > P̄(t), sinon 0, avec P = −Σ_i (z_i − f_i)². Le taux η(t) décroît avec le temps. Le texte principal ne contient **aucune normalisation** du pas, ni par ‖r‖², ni par la variance du bruit, ni par l'amplitude de M.

### Cited Findings
- **Référence.** Hoerzer G.M., Legenstein R., Maass W., « Emergence of complex computational structures from chaotic neural networks through reward-modulated Hebbian learning », *Cerebral Cortex* 24(3):677–690, 2014 (publié en ligne le 13 novembre 2012, doi:10.1093/cercor/bhs348). C'est pourquoi on le trouve cité « Hoerzer et al. 2012 ». **[lu]**, PDF auteur (identique à igi-web.tugraz.at/PDF/214.pdf), copie locale `~/Documents/Jabref/PhD/Hoerzer2012b.pdf`. — [Cerebral Cortex](https://academic.oup.com/cercor/article/24/3/677/394650) ; [PDF TU Graz](https://igi-web.tugraz.at/PDF/214.pdf)
- **Réservoir.** N = 1000 neurones à taux, connectivité 0,1, τ ẋ = −x + λ W^rec r + W^in u + W^fb z. Les taux sont r = tanh(x + ξ^state). λ place le réseau dans le régime chaotique avant l'apprentissage, comme dans Sussillo & Abbott 2009. **[lu]** — [PDF](https://igi-web.tugraz.at/PDF/214.pdf)
- **Readout.** ẑ_i(t) = w_iᵀ r(t) + b_i, linéaire et sans tanh, « to match the model of Sussillo and Abbott (2009) ». La sortie bruitée est z_i = ẑ_i + ξ_i(t), avec un bruit d'exploration de moyenne nulle. Le z bruité est renvoyé au réseau. Le bruit peut être retiré au test, avec une performance similaire. Le biais est gardé « for consistency with Legenstein et al. (2010) », même si les tâches n'en ont pas besoin. **[lu]** — [PDF](https://igi-web.tugraz.at/PDF/214.pdf)
- **Signal de modulation.** Équation (5) : M(t) = 1 si P(t) > P̄(t), 0 sinon. P̄ est un filtre passe-bas de P, dont les détails sont dans les Méthodes supplémentaires. Équation (6) : la performance est la somme des erreurs quadratiques de **tous** les readouts, avec un signe négatif ; ce signe est perdu dans l'extraction du texte mais confirmé par Pogodin et al., qui écrivent P = −‖f − z‖². Ce M unique module **toutes** les synapses de **tous** les readouts : un signal global, pas un signal par sortie. **[lu]** — [PDF](https://igi-web.tugraz.at/PDF/214.pdf) ; [Pogodin et al. 2019](https://arxiv.org/html/1910.10559)
- **Mise à jour.** Équation (7) : Δw_i(t) = η(t)[z_i(t) − z̄_i(t)] M(t) r(t). z̄_i est un filtre passe-bas de la sortie bruitée. Si ẑ varie peu à l'échelle du filtre et si le bruit est peu corrélé dans le temps, z − z̄ ≈ ξ. La règle n'a donc pas besoin de connaître le bruit, contrairement à Fiete & Seung 2006. **[lu]** — [PDF](https://igi-web.tugraz.at/PDF/214.pdf)
- **Écarts avec l'EH original de Legenstein et al. (2010, J. Neurosci.).** Le M original est analogique : M = P − P̄. Hoerzer le rend binaire, donc moins informatif. Selon les résultats supplémentaires, la performance du M binaire est « comparable » à celle du M analogique. **[lu]** — [PDF](https://igi-web.tugraz.at/PDF/214.pdf) ; [Legenstein et al. 2010](https://motorlab.pitt.edu/pub/reward-modulated.pdf) (non lu)
- **Taux d'apprentissage.** η(t) est « a small learning rate that can either be constant, or decay over time ». Toutes les simulations de l'article utilisent un taux décroissant. Les η de EH et de LMS-FORCE ont été choisis séparément, par une recherche exhaustive (brute force). **[lu]** — [PDF](https://igi-web.tugraz.at/PDF/214.pdf)
- **Valeurs numériques**, d'après Pogodin et al. 2019, qui reprennent le cadre de Hoerzer.
  - η(t) = η₀/(1 + t/τ_η), avec η₀ = 5·10⁻⁴ et τ_η = 2·10⁴ ms, soit 20 s.
  - Bruit d'exploration uniforme sur [−0,5 ; 0,5].
  - Filtres passe-bas : x̄(t+dt) = x̄ + dt (x − x̄)/τ_f, avec τ_f = 5 ms. Pour dt = 1 ms, cela donne α = 0,2.
  - Poids initiaux nuls ; λ = 1,5 ; τ = 50 ms ; p = 0,1 ; N = 1000.

  Ces valeurs concordent avec la note locale du projet (`eh_numpy.py` vérifié comme fidèle : `learning_rate=5e-4`, `decay=20000`, `noise=0.5`, `alpha=0.2`, `reward="global"`). **[méthodes lues]** — [Pogodin et al. 2019](https://arxiv.org/html/1910.10559)
- **Conditions nécessaires**, étudiées sur 45 réglages et 50 essais chacun.
  - Le signal cible doit être lent : composantes ≤ 4 Hz pour τ = 50 ms. Réduire τ aide pour les cibles plus rapides.
  - Les mises à jour de M et de w doivent être fréquentes : 1 ms convient ; 5 et 10 ms ne conviennent que pour des cibles lentes.
  - Un bruit d'exploration corrélé dans le temps (τ_noise de 1 à 5 ms) dégrade l'apprentissage.

  **[lu]** — [PDF](https://igi-web.tugraz.at/PDF/214.pdf)
- **Tâches de l'article.**
  - Génération d'un motif périodique de 4 sinusoïdes (période 1 s) avec rétroaction de la sortie.
  - Mémoire de travail : 2 readouts avec un M commun, entrées on/off.
  - Routage d'information dépendant de l'état : 3 readouts.
  - Calcul analogique non linéaire.

  **[lu]** pour les deux premières ; les deux dernières d'après le texte et les légendes — [PDF](https://igi-web.tugraz.at/PDF/214.pdf)

### Inferences
- **Échelle de η.** Avec r = tanh(·) ∈ [−1, 1]ᴺ et N = 1000, ‖r‖² vaut quelques centaines. Le pas effectif dans l'espace de sortie, de l'ordre de η‖r‖², est donc d'environ 0,05 à 0,5 pour η = 5·10⁻⁴. La valeur de η₀ est implicitement liée à N et à l'échelle des entrées. Il faut la réajuster si l'on change N ou si l'on passe à des caractéristiques de norme plus grande (monômes, concaténation [r, r²]). C'est précisément le problème que corrige une normalisation de type NLMS.
- **Échelle de M.** Le M binaire borne la modulation à {0, 1}. Le pas ne dépend donc ni de l'amplitude de l'erreur ni de l'échelle de la récompense. C'est la seule « normalisation » implicite de la règle. Le M analogique P − P̄ réintroduit une dépendance à l'échelle de l'erreur.
- **Biais de la règle.** M ≥ 0 (pas de mise à jour négative). Le terme z − z̄ n'est donc pas centré sur la seule exploration quand ẑ dérive. C'est pourquoi la règle exige une entrée et une cible lentes : c'est l'hypothèse sur ẑ qui justifie z − z̄ ≈ ξ.

### Gaps
- Les Méthodes supplémentaires ([214_suppl.pdf](https://igi-web.tugraz.at/PDF/214_suppl.pdf)) sont restées inaccessibles : délai dépassé et erreur SSL. La constante de temps exacte du filtre de P̄ dans Hoerzer lui-même, les valeurs de η trouvées par la recherche exhaustive et la dynamique de décroissance n'ont donc pas été lues directement. Elles proviennent de Pogodin et al. et de la note locale.

## Q2. Papiers qui citent et modifient EH : inventaire

### Takeaway
Les descendants directs modifient surtout quatre choses : le signal de modulation (analogique, sous-linéaire, avec trace d'éligibilité), l'amplitude de l'exploration (modulée par l'erreur), l'entrée du réservoir (attracteur de mémoire de travail, oscillateurs modulaires) et l'architecture (double voie avec un FORCE « maître »). Aucun ne normalise le pas par ‖r‖². Plusieurs signalent que EH **ne converge pas** (poids oscillants) ou échoue sur des cibles longues ou rapides.

### Cited Findings
- **Pyle & Rosenbaum 2019, *Neural Computation* 31(7)** (doi:10.1162/neco_a_01198 ; prépublication arXiv 1803.03304, 2018). **[méthodes lues]**, PDF arXiv.
  - Version « RMHL » modifiée : erreur scalaire quelconque e(z, t) ≥ 0, pas forcément une cible.
  - Sortie z = W r + Ψ(e) η : l'**amplitude du bruit dépend de l'erreur**, avec Ψ sous-linéaire croissante et Ψ(0) = 0.
  - Mise à jour τ_w dW/dt = Φ(ê) ẑ rᵀ. ê et ẑ sont des filtres passe-haut (x̂ = x − x̄, τ = 1 ms) ; Φ est impaire et sous-linéaire, par exemple Φ(x) = −5·x^{1/4}, et Ψ(x) = 0,025·(10x)^{1/4}.
  - Paramètres : τ_w = 0,02, dt = 0,2 ms, N = 1000, λ = 1,5, bruit uniforme sur [−1, 1].
  - L'exploration et l'apprentissage s'éteignent quand e → 0. C'est une **adaptation du pas et de l'exploration par l'erreur**, pas par l'entrée.
  - Le papier propose SUPERTREX : une voie exploratoire RMHL plus une voie « maîtrise » W₂ apprise par FORCE/RLS sur z₁, avec dW₂/dt = −k z₁ rᵀ P, et un seuil doux sur e. On peut remplacer P par I, ce qui donne LMS : c'est plus lent, mais la règle devient locale.
  - Résultat : « the weights trained by RMHL fail to converge » ; sur le dessin d'un papillon, ‖W‖ oscille et RMHL imite la sortie par des changements rapides de W.

  — [arXiv 1803.03304](https://arxiv.org/abs/1803.03304) ; [par.nsf.gov](https://par.nsf.gov/biblio/10098034)
- **Sankar et al. 2021, ReScience C 7(11)**, « [Re] A Reservoir Computing Model of Reward-Modulated Motor Learning and Automaticity » : ré-implémentation Python de SUPERTREX et RMHL. **[résumé]** — [code](https://github.com/rsankar9/Reimplementation-SUPERTREX/releases/tag/v3.0)
- **Pogodin, Corneil, Seeholzer, Heng & Gerstner 2019, atelier NeurIPS « Real Neurons & Hidden Units »** (arXiv 1910.10559). **[méthodes lues]**
  - Règle de Hoerzer telle quelle, sous forme matricielle : ΔW = η(t) M(t)(z − z̄) rᵀ, M binaire, P = −‖f − z‖², avec les valeurs de Q1.
  - Ajout d'un réseau attracteur continu (2500 neurones, tore 50 × 50, adaptation) qui fournit au réservoir un « spatio-temporal backbone signal ».
  - Variante : les mises à jour sont accumulées et appliquées en fin d'essai de 1 s, avec η constant.
  - Aucune normalisation ni écrêtage mentionné.

  — [arXiv HTML](https://arxiv.org/html/1910.10559)
- **Burms, Caluwaerts & Dambre 2015, *Frontiers in Neurorobotics* 9:9**, « Reward-Modulated Hebbian Plasticity as Leverage for Partially Embodied Control in Compliant Robotics ». **[méthodes lues]**, page HTML dont les équations n'ont pas été rendues.
  - Règle de type RMH avec bruit **explicite** injecté en postsynaptique, et non estimé par z − z̄ comme dans EH. L'espérance de la récompense est soustraite, soit par une moyenne sur 50 essais par entrée, soit par RLS, soit sur 30 essais par caractère.
  - Variante Heaviside : mise à jour seulement si la récompense dépasse l'attente, l'analogue du M binaire.
  - **Décorrélation des entrées** à chaque essai par pseudo-inverse de Moore-Penrose (régularisation λ = 1), une forme de pré-conditionnement ou blanchiment de l'entrée.
  - Contrainte optionnelle sur la norme des poids, qui ramène à une règle proche de celle de Sanger.
  - Appliqué à un réservoir physique (robot de tenségrité, écriture de caractères) et à des réseaux simulés : XOR à 2 bits retardé, décodeur à 3 bits, observation non linéaire.
  - Résultats : l'estimation du bruit à la Legenstein (EH) est « severely worse » sur le décodeur à 3 bits. La version décorrélée résiste au bruit corrélé, contrairement à la règle par défaut. CMA-ES ne converge pas en 300 000 essais sur le XOR (≈ 10⁴ dimensions).

  — [Frontiers](https://www.frontiersin.org/articles/10.3389/fnbot.2015.00009/full)
- **Kumar, Tan, Libedinsky, Yen & Tan 2021**, arXiv 2106.03580, « One-shot learning of paired associations by a reservoir computing model with Hebbian plasticity » (v5 en 2026). **[méthodes lues]**, PDF v1.
  - EH « à 4 facteurs » : g_noisy = g + σ√α·N(0, 1), avec σ = 0,25.
  - P = −‖p − g_noisy‖² ; M binaire ; filtres P̄ et ḡ remis à 0 à chaque essai.
  - ΔW = r (g_noisy − ḡ) R M : le 4ᵉ facteur est la délivrance de la récompense R, qui sert de porte.
  - η = 5·10⁻⁴, contre 0,05 pour la règle perceptron.
  - Tâche de navigation avec associations indice-lieu en un essai. Aucune normalisation de l'entrée.

  — [arXiv](https://arxiv.org/abs/2106.03580)
- **Kawai & Asada 2023, *Neurocomputing*** (doi:10.1016/j.neucom.2023.126740), « Spatiotemporal motor learning with reward-modulated Hebbian plasticity in modular reservoir computing ». **[résumé]**
  - RMHL (lignée Hoerzer et Pyle & Rosenbaum) sur un réservoir modulaire « reBASICS » : petits réseaux aux oscillations de cycle limite auto-entretenues, de fréquences variées, intégrées linéairement.
  - Réussit le timing moteur et le dessin de motifs, là où le RMHL sur un réservoir classique échoue.
  - Une **trace d'éligibilité** permet d'apprendre avec une récompense retardée.

  — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231223008639) ; [OUCI](https://ouci.dntb.gov.ua/works/4OwMzoO4)
- **Matsuki & Shibata**. **[résumé]**
  - ICONIP 2016 (LNCS) : « Reward-Based Learning of a Memory-Required Task Based on the Internal Dynamics of a Chaotic Neural Network ».
  - Article de 2020 dans *Neural Networks*, « Adaptive balancing of exploration and exploitation around the edge of chaos in internal-chaos-based learning » : l'exploration vient de la **dynamique chaotique interne** au lieu du bruit externe de Hoerzer. La chaoticité baisse pendant l'apprentissage, d'où un passage automatique de l'exploration à l'exploitation.
  - Suite en 2024 : Matsuki, Sakemi & Aihara, arXiv 2405.09086, avec TD3 et un ESN.

  — [Springer](https://www.springerprofessional.de/en/reward-based-learning-of-a-memory-required-task-based-on-the-int/10799684) ; [AMiner 2020](https://www.aminer.org/pub/5f4cd1f39fced0a24bedba8b/adaptive-balancing-of-exploration-and-exploitation-around-the-edge-of-chaos-in) ; [arXiv 2405.09086](https://arxiv.org/pdf/2405.09086v1.pdf)
- **Aswolinskiy & Pipa 2015, *Front. Comput. Neurosci.* 9:36**, « RM-SORN ». Cite Hoerzer : son bruit d'exploration y est remplacé par la plasticité intrinsèque, qui est déterministe. Neurones binaires à seuil, R-STDP sur le readout. La soustraction de base M_k est décrite comme « similar » à celle de Hoerzer. **[méthodes lues]** — [Frontiers](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00036/full)
- **Daruwalla & Lipasti**, « Information bottleneck-based Hebbian learning rule » (arXiv 2111.13187, 2021 ; *Front. Comput. Neurosci.* 18:1240348, 2024). **Sources contradictoires** sur la règle du réseau auxiliaire, réservoir de mémoire de travail :
  - selon un extrait arXiv, il est entraîné par une règle hebbienne à 3 facteurs « as stated in Hoerzer, Legenstein, and Maass (2014) » ;
  - selon la version Frontiers, il est entraîné par un FORCE LMS adapté de Sussillo & Abbott.

  Le corps du papier (MNIST > 91 %, CIFAR-10 ≈ 61 %) utilise Adam. **[résumé]** — [arXiv](https://arxiv.org/pdf/2111.13187) ; contredit par [Frontiers 2024](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1240348/full)
- **García Fernández, Ahmad & van Gerven**, arXiv 2503.23972 (2025, v4 en 2026), « Noise-based reward-modulated learning » (NRL). **[méthodes lues]**
  - Leur RMHL (Legenstein 2010 / Miconi 2017) utilise un bruit **explicite**, la version qui infère le bruit à partir de l'activité étant « too unstable ».
  - NRL : bruit gaussien sur les pré-activations de chaque couche, éligibilité Σ ξ̄ ρ x̃ᵀ avec **ξ̄ = ξ/‖ξ‖²**, ce qui normalise par la norme du bruit et non de l'entrée.
  - RPE δ = r − r̄, avec λ = 0,66. Sur Acrobot et Cartpole, la RPE est divisée par le retour.

  — [arXiv HTML](https://arxiv.org/html/2503.23972v4)
- **Yada, Yasuda & Takahashi 2021, *Applied Physics Letters*** (doi:10.1063/5.0064771), « Physical reservoir computing with FORCE learning in a living neuronal culture ». C'est **FORCE**, pas EH : culture neuronale vivante comme réservoir, robot dans un labyrinthe, retour par stimulation « homéostatique ». **[résumé]**, communiqué de presse. — [EurekAlert](https://www.eurekalert.org/news-releases/932479)
- **Legenstein et al. 2010, *J. Neurosci.*** : origine de EH, avec M analogique P − P̄ et une tâche d'interface cerveau-machine. Non lu ici. — [PDF](https://motorlab.pitt.edu/pub/reward-modulated.pdf)

### Inferences
- La lignée « réservoir + EH » est petite : environ 6 à 8 papiers trouvés qui reprennent la règle telle quelle ou la modifient. Les modifications portent sur la modulation et l'exploration, jamais sur la géométrie du pas dans l'espace des poids.
- Les deux remèdes à la non-convergence sont différents :
  - Pyle & Rosenbaum confient la consolidation à un FORCE/RLS (SUPERTREX) ;
  - Pogodin et Kawai enrichissent l'entrée du réservoir (attracteur, oscillateurs) pour que des états distincts et répétables rendent le problème mieux conditionné.

  Dans les deux cas, le conditionnement de l'entrée paraît déterminant. C'est cohérent avec la décorrélation de Burms et al. et avec l'intérêt d'un pas normalisé ou pré-conditionné.

### Gaps
- Je n'ai pas obtenu de liste « cited by » complète (Google Scholar ou Semantic Scholar non consultés). L'inventaire repose sur des recherches par mots-clés : des applications en robotique, en interfaces cerveau-machine ou en neuromorphique ont pu m'échapper.
- Je n'ai trouvé aucun travail ultérieur de Legenstein reprenant EH sur des readouts de réservoir, ni aucune implémentation EH en impulsions ou sur matériel neuromorphique (Loihi, SpiNNaker, memristors).
- Je n'ai pas vérifié de « Kawai et al. » autre que Kawai & Asada 2023.
- Le texte intégral de Kawai & Asada 2023 est inaccessible (403), donc ni ses équations ni la forme de sa trace d'éligibilité n'ont été lues.

## Q3. EH normalisé (pas de type NLMS) et EH sur caractéristiques non linéaires, noyaux, monômes ou readouts profonds

### Takeaway
Je n'ai trouvé **aucun** « EH normalisé » au sens η/(‖x‖² + ε), ni aucune application de EH à des caractéristiques polynomiales, à des monômes ou à des noyaux de l'état du réservoir. Les « normalisations » existantes portent sur d'autres grandeurs :
- l'amplitude de l'erreur (Pyle & Rosenbaum : Ψ(e), Φ(ê) sous-linéaires) ;
- le binaire M ∈ {0, 1} (Hoerzer) ;
- la norme du bruit (García Fernández : ξ/‖ξ‖²) ;
- la décorrélation des entrées par pseudo-inverse (Burms et al.) ;
- la norme des poids (Burms et al.).

Sur des réseaux profonds, le seul résultat trouvé est que RMHL, avec bruit explicite, peine à attribuer le crédit au-delà d'une couche.

### Cited Findings
- Hoerzer et al. : aucune normalisation dans l'équation (7). η(t) n'est que constant ou décroissant dans le temps, et M est binaire. Le readout est linéaire sur r = tanh(x), sans caractéristiques non linéaires supplémentaires. **[lu]** — [PDF](https://igi-web.tugraz.at/PDF/214.pdf)
- Pyle & Rosenbaum : adaptation du pas par l'erreur, avec dW ∝ Φ(ê) ẑ rᵀ et un bruit ∝ Ψ(e), ce qui éteint l'exploration quand e → 0. Pas de normalisation par ‖r‖². **[méthodes lues]** — [arXiv](https://arxiv.org/abs/1803.03304)
- Pogodin et al. : « None » (aucune normalisation ni écrêtage mentionné) ; η₀/(1 + t/τ_η). **[méthodes lues]** — [arXiv HTML](https://arxiv.org/html/1910.10559)
- Burms et al. 2015 : observations **décorrélées** par pseudo-inverse régularisée à chaque essai, plus une contrainte de norme sur les poids. C'est la seule normalisation côté entrée trouvée, mais elle relève du blanchiment et non d'une division par ‖x‖². **[méthodes lues]** — [Frontiers](https://www.frontiersin.org/articles/10.3389/fnbot.2015.00009/full)
- García Fernández et al. :
  - éligibilité normalisée par ‖ξ‖² (ξ̄ = ξ/‖ξ‖²) et RPE divisée par le retour ;
  - réseaux à 2 et 3 couches cachées (64 à 128 unités) sur Acrobot, Cartpole et une tâche d'atteinte : RMHL « struggles with credit assignment in deeper networks », avec des « minimal performance improvements across trials » ;
  - NRL approche la rétropropagation, mais plus lentement quand la profondeur augmente ;
  - η et σ de RMHL : 5e-2/1e-3 pour Acrobot, 1e-2/1e-1 pour Cartpole, 1e-1/1e-1 pour l'atteinte.

  **[méthodes lues]** — [arXiv HTML](https://arxiv.org/html/2503.23972v4)
- Hiratani et al., NeurIPS 2022, « On the stability and scalability of node perturbation learning » : cité comme analyse de la stabilité et du passage à l'échelle des règles par perturbation. Contenu non lu. **[résumé]**, citation seulement. — [arXiv 2503.23972 (bibliographie)](https://arxiv.org/html/2503.23972v4)
- Les recherches ciblées « normalized reward-modulated Hebbian », « NLMS » + « reward-modulated » et « EH » + « polynomial/kernel features » n'ont rien donné de pertinent. — [recherche web, aucune source]

### Inferences
- Dans EH, le changement de sortie produit par une mise à jour vaut Δẑ = Δwᵀ r ∝ η ‖r‖² (z − z̄) M. Sans normalisation, la taille effective du pas varie d'un pas de temps à l'autre avec ‖r(t)‖². Elle change aussi d'échelle quand on étend les caractéristiques : pour φ(r) = [r, r ⊙ r, …], ‖φ‖² augmente et η₀ = 5·10⁻⁴, réglé pour N = 1000, devient trop grand ou trop petit.
- La normalisation η/(‖φ‖² + ε) rendrait le pas invariant à l'échelle et à la dimension des caractéristiques. C'est exactement l'analogue NLMS de l'EH « LMS-like ». Hoerzer montre en effet qu'EH se comporte comme LMS-FORCE (Q4).
- Sur des monômes, la corrélation entre composantes augmente le nombre de conditionnement de E[φφᵀ]. EH comme LMS convergent alors lentement dans les directions de faible variance. Les remèdes publiés (décorrélation de Burms, RLS de SUPERTREX) agissent sur ce conditionnement. Une normalisation scalaire de type NLMS corrige l'échelle mais pas la corrélation.
- C'est un **trou dans la littérature**. Un « EH normalisé » sur les monômes de l'état serait, d'après cette recherche, une contribution nouvelle.

### Gaps
- Je n'ai pas consulté de liste exhaustive « cited by » dans Semantic Scholar ou Google Scholar : une variante normalisée publiée dans un actes de conférence peu indexé reste possible.
- Le contenu de Hiratani et al. 2022 n'a pas été lu, notamment la question de savoir s'ils normalisent le pas de la node perturbation.
- La tâche 3 de Burms et al. (« non-linear observation ») n'a pas pu être précisée : il n'est pas établi qu'elle utilise des caractéristiques non linéaires du readout.

## Q4. Comparaisons de EH avec FORCE (RLS), LMS et ridge

### Takeaway
La seule comparaison contrôlée de l'article d'origine donne RLS-FORCE ≫ EH ≈ LMS-FORCE. Sur le motif périodique, RLS-FORCE est bon dès 10 s d'apprentissage ; EH et LMS demandent environ 100 s et ne diffèrent pas significativement au-delà de 100 s ; les η sont réglés séparément pour chaque règle. Les travaux suivants confirment que EH est derrière FORCE et souvent non convergent. Je n'ai trouvé **aucune comparaison de EH avec un readout ridge (hors ligne)**.

### Cited Findings
- **Hoerzer et al., figure 1G.**
  - 50 essais par durée d'apprentissage (10 à 400 s), test sur 500 s.
  - RLS-FORCE est le meilleur : bonne approximation après 10 s d'apprentissage (10 cycles), « not surprising since the RLS-based rule uses nonlocal information ».
  - EH se comporte comme LMS-FORCE : environ 100 s nécessaires, pas de différence significative au-delà de 100 s (test de Wilcoxon, 5 %).
  - La performance de EH reste stable au-delà de 150 s.
  - Les η de EH et de LMS ont été fixés séparément par recherche exhaustive.

  **[lu]** — [PDF](https://igi-web.tugraz.at/PDF/214.pdf)
- **Pogodin et al.**
  - Cibles de processus gaussiens de 1 s et 10 s ; 50 cibles × 50 réservoirs.
  - FORCE atteint une erreur de test proche de 1, mesurée en corrélation croisée, après 30 à 50 essais.
  - rmHebb avec entrée attracteur égale FORCE après environ 90 essais sur 1 s, et atteint 1 après environ 200 essais sur 10 s ; FORCE atteint 1 dans tous les essais.
  - rmHebb sans attracteur échoue sur 10 s, même avec un réglage des hyperparamètres en échelle log.
  - La règle standard ne tolère que de très petits délais : elle échoue si les mises à jour sont plus espacées que quelques ms.
  - MNIST (trajectoires de chiffres) : les 10 chiffres sont appris avec un attracteur sans bruit.

  **[méthodes lues]** — [arXiv HTML](https://arxiv.org/html/1910.10559)
- **Pyle & Rosenbaum.**
  - Tâche 1 (papillon) : FORCE est meilleur pendant l'apprentissage ; RMHL imite la cible mais ne converge pas (‖W‖ oscille) ; SUPERTREX converge et fait mieux au test qu'à l'apprentissage.
  - Tâches 2 et 3 (bras à 2 ou 3 articulations, erreur dans l'espace de la tâche) : FORCE n'est pas applicable ; RMHL « performed poorly » ; SUPERTREX réussit.

  **[méthodes lues]** — [arXiv](https://arxiv.org/abs/1803.03304)
- **Burms et al.** : RMH avec bruit explicite contre EH avec bruit estimé : EH est « severely worse » sur le décodeur à 3 bits. RMH contre CMA-ES : RMH trouve une solution là où CMA-ES ne converge pas en 3·10⁵ essais. **[méthodes lues]** — [Frontiers](https://www.frontiersin.org/articles/10.3389/fnbot.2015.00009/full)
- **Kawai & Asada 2023** : les résultats annoncés concernent RMHL sur reBASICS contre RMHL sur un réservoir standard, mais aucun chiffre n'a été lu. **[résumé]** — [OUCI](https://ouci.dntb.gov.ua/works/4OwMzoO4)
- **García Fernández et al.** : RMHL contre NRL contre rétropropagation sur des tâches de renforcement ; RMHL est le moins bon dans les réseaux profonds. Les chiffres ne figurent que dans les graphiques. **[méthodes lues]** — [arXiv HTML](https://arxiv.org/html/2503.23972v4)

### Inferences
- EH est, au mieux, l'équivalent par récompense d'un LMS. Les comparaisons équitables publiées règlent η séparément pour chaque règle, mais toujours sur le **même** réservoir, fixé par Sussillo & Abbott, sans optimisation par règle. Elles ne satisfont donc pas le critère « réservoir optimisé pour chaque configuration » de ce projet.
- Puisque EH ≈ LMS dans la seule comparaison directe, une normalisation qui améliore LMS, comme NLMS, est un candidat naturel pour améliorer EH. Ce n'est pas testé dans la littérature.

### Gaps
- Je n'ai trouvé aucune comparaison de EH avec ridge (régression de Tikhonov hors ligne) ou avec NLMS sur des benchmarks de réservoir (NARMA, Mackey-Glass, classification).
- Je n'ai trouvé aucune comparaison sur des données de classification réelles hors MNIST : Pogodin, et Daruwalla où la règle réelle n'est pas certaine.

## Q5. Implémentations disponibles (bibliothèques et code)

### Takeaway
reservoirpy ne contient **pas** de nœud EH, ni dans la version officielle ni dans la branche locale `ridge-solvers` basée sur v0.4.3-dev. Il n'y a que LMS, RLS, Ridge et les plasticités non supervisées. Les implémentations d'EH sont du code de recherche isolé, en ANNarchy ou en Python, plus l'implémentation locale de canopy.

### Cited Findings
- **Local.**
  - `~/Code/Autres/reservoirpy` (branche `ridge-solvers`) : `reservoirpy/nodes/` contient `lms.py`, `rls.py`, `ridge.py`, `local_plasticity_reservoir.py`, `intrinsic_plasticity.py`, etc. Aucun fichier ne mentionne « hoerzer », « reward » ou « exploratory », et `lms.py` ne normalise pas le pas.
  - Le projet a sa propre implémentation, `canopy/models/readout_rules/eh_numpy.py`, un `OnlineNode` reservoirpy. Options : `reward` par sortie ou global, `modulation` binaire ou continue, `baseline` filtrée ou par échantillon (Fiete & Seung), décroissance η₀/(1 + t/decay). Elle n'a pas de normalisation par ‖x‖² non plus.

  **[lu]**, inspection locale. — fichiers locaux
- **esyniawa/rc-ANN_rm_hebbian** : réservoir entraîné par la règle hebbienne modulée de Legenstein et al. 2010 et Hoerzer et al. 2014, en ANNarchy, piloté depuis Python. Trouvé seulement sur un miroir tiers ; le dépôt github.com canonique n'a pas été vérifié. **[résumé]** — [miroir](https://github.xiaohuzi123231346.workers.dev/esyniawa/rc-ANN_rm_hebbian)
- **Code de Pogodin et al.** (dépôt anonymisé de l'atelier NeurIPS). **[résumé]** — [GitHub](https://github.com/neuroai-workshop-anon-1224113/working-memory-facilitating-reservoir-learning)
- **Ré-implémentation ReScience de SUPERTREX/RMHL** (Sankar et al. 2021), en Python. **[résumé]** — [GitHub](https://github.com/rsankar9/Reimplementation-SUPERTREX/releases/tag/v3.0)
- **hebbRNN** (Michaels & Scherberger, JOSS 2016) : règle hebbienne modulée par la récompense pour réseaux récurrents, de type Miconi, pas EH sur un readout ; langage non vérifié. **[résumé]** — [GitHub](https://github.com/jonathanamichaels/hebbrnn)
- **Daruwalla & Lipasti** : code de la règle HSIC avec réservoir de mémoire de travail. **[résumé]** — [GitHub](https://github.com/darsnack/biological-hsic)

### Inferences
- Ajouter une option `normalize` (η/(‖x‖² + ε)) à `eh_numpy.py` serait trivial. Ce serait, d'après cette recherche, la première implémentation publique d'un « EH normalisé ».

### Gaps
- Je n'ai pas vérifié l'absence d'EH dans d'autres bibliothèques de RC : PyRCN, EchoTorch, ReservoirComputing.jl, Brian2 et NEST (exemples).
- Le code original de Hoerzer et al. n'a pas été trouvé ; je n'ai pas vérifié s'il existe une entrée ModelDB.
