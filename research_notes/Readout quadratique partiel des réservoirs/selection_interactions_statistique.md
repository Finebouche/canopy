# Sélection de termes d'interaction x_i x_j en grande dimension : littérature statistique et apprentissage (hors reservoir computing)

Convention de vérification utilisée dans toutes les sections :

- **[V]** : affirmation lue dans la source elle-même (résumé arXiv ou texte intégral via ar5iv / PMC).
- **[S]** : affirmation tirée d'une source secondaire (résumé de moteur de recherche, article qui cite, page de dépôt).
- Les éléments connus seulement de mémoire, sans source ouverte, sont rangés dans les rubriques « Gaps ».

Notations : x ∈ R^p est le vecteur des variables, Σ = Cov(x), Z_ij = x_i x_j, r est le résidu du modèle linéaire, H̄ = E[∂²E(y|x)/∂x∂xᵀ] est la hessienne moyenne de la fonction de régression.

Deux corrections de références par rapport à la demande :

- « High-dimensional interaction detection with false sign rate control » n'est pas de Tang, Fang et Dong. Les auteurs sont Daoji Li, Yinfei Kong, Yingying Fan et Jinchi Lv, J. Business & Economic Statistics 40(3):1234-1245, 2022, DOI 10.1080/07350015.2021.1917419 [S] ([EconPapers](https://econpapers.repec.org/RePEc:taf:jnlbes:v:40:y:2022:i:3:p:1234-1245)).
- L'article de Tang, Fang et Dong (2020) est « High-dimensional interactions detection with sparse principal Hessian matrix », JMLR 21(19):1-25 [S] ([JMLR](https://www.jmlr.org/papers/v21/19-071.html)).

---

## 1. Principe d'hérédité / de marginalité : méthodes, hypothèses, et ce qu'elles disent de sa violation

### Takeaway

L'hérédité est une hypothèse de *structure creuse dans les coordonnées d'origine* : elle est justifiée par l'interprétabilité et par la parcimonie pratique, elle dépend de la paramétrisation, et les bornes minimax montrent qu'elle n'améliore pas toujours la vitesse d'estimation. Tous les articles qui la défendent supposent un petit nombre d'interactions actives ; aucun ne traite un signal quadratique diffus sur des dizaines de milliers de produits.

### Cited Findings

- **Définitions.** L'hérédité forte impose γ_jk ≠ 0 seulement si β_j β_k ≠ 0 ; l'hérédité faible exige un seul parent actif. Les noms « strong » et « weak heredity » viennent de Chipman (1996), le principe de marginalité de Nelder (1977), et l'« effect heredity » des plans d'expériences de Hamada et Wu (1992) [S] ([Hao & Zhang, arXiv:1412.7138](https://arxiv.org/abs/1412.7138) ; [Yuan, Joseph & Zou 2009, Ann. Appl. Stat., DOI 10.1214/09-AOAS254](https://ar5iv.labs.arxiv.org/html/1011.0610)).
- **Bien, Taylor & Tibshirani (2013), « A lasso for hierarchical interactions », Annals of Statistics 41(3):1111-1141, DOI 10.1214/13-AOS1096 (hierNet).** Le lasso reçoit des contraintes convexes qui garantissent la hiérarchie avec probabilité un ; l'article distingue la parcimonie des paramètres de la « practical sparsity » (nombre de variables brutes à mesurer) et donne une estimation sans biais des degrés de liberté [V, résumé] ([arXiv:1205.5050](https://arxiv.org/abs/1205.5050)). Le paquet couvre les versions forte et faible [S] ([CRAN hierNet](https://cran.r-project.org/web/packages/hierNet/index.html)).
- **Lim & Hastie, « Learning interactions through hierarchical group-lasso regularization » (glinternet), arXiv:1308.2719.** La pénalité group-lasso avec recouvrement impose l'hérédité forte. L'argument donné est qu'il « rarely make sense to have interactions without main effects ». Les simulations couvrent une vérité hiérarchique et une vérité *anti-hiérarchique* : glinternet est comparable à hierNet quand la vérité est hiérarchique et meilleur quand elle est anti-hiérarchique, parce que hierNet utilise les effets principaux « as a budget for interactions ». Le boosting d'arbres n'est pas compétitif dans le cas anti-hiérarchique. hierNet est limité en pratique à moins de 1 000 variables [V, texte ar5iv] ([ar5iv 1308.2719](https://ar5iv.labs.arxiv.org/html/1308.2719)).
- **Hao & Zhang (2014), « Interaction screening for ultrahigh-dimensional data », JASA 109(507):1285-1301, DOI 10.1080/01621459.2014.881741 (iFORT, iFORM).** La sélection ascendante gloutonne ajoute effets principaux et interactions en maintenant la hiérarchie : une interaction n'est candidate qu'une fois ses parents entrés. Les auteurs prouvent la propriété de « sure screening » en ultra-haute dimension pour des modèles creux [S, résumé de l'éditeur] ([IDEAS/RePEc](https://ideas.repec.org/a/taf/jnlasa/v109y2014i507p1285-1301.html)).
- **Hao, Feng & Zhang, « Model selection for high-dimensional quadratic regression via regularization » (RAMP), arXiv:1501.00049.** L'article étudie les méthodes lasso en deux étapes (effets principaux, puis interactions entre les effets retenus), en établit les propriétés théoriques, et propose RAMP, qui calcule un chemin de régularisation respectant la hiérarchie [V, résumé] ([arXiv:1501.00049](https://arxiv.org/abs/1501.00049)).
- **Hao & Zhang, « A note on high dimensional linear regression with interactions », arXiv:1412.7138.** Le fait qu'un modèle satisfasse l'hérédité « does depend on its parametrization ». L'importance d'un effet est définie de façon invariante seulement pour les recodages x̃_j = a_j (x_j − c_j). Les modèles qui violent l'hérédité forte forment un ensemble de mesure nulle de l'espace des paramètres. Les méthodes en deux étapes sont justifiées quand la covariance de (X, Z) est bloc-diagonale, ce qui est vrai si la loi de X centré est symétrique (moments d'ordre 1 et 3 nuls). Le contre-exemple de Turlach (2004), Y = (X₁ − 0,5)² + X₂ + … + X₅ + ε avec X uniforme, annule cov(Y, X₁) et fait échouer l'étape 1 ; les auteurs le jugent « too special to be representative » [V, texte ar5iv] ([ar5iv 1412.7138](https://ar5iv.labs.arxiv.org/html/1412.7138)).
- **Ye & Yang, « High-dimensional adaptive minimax sparse estimation with interactions », arXiv:1804.02482.** Les vitesses minimax sont établies sous hérédité forte, faible et sans hérédité. Une condition d'hérédité plus forte « may or may not drastically improve the minimax rate of convergence » ; dans certains régimes la vitesse est identique dans les trois cas, et le coût d'estimation des interactions domine souvent [V, résumé] ([arXiv:1804.02482](https://arxiv.org/abs/1804.02482)).
- **She, Wang & Jiang (2018), « Group regularized estimation under structural hierarchy », JASA 113(521):445-454, DOI 10.1080/01621459.2016.1260470.** Plusieurs pénalités de groupe imposent la hiérarchie ; l'article donne des inégalités oracle et des bornes minimax inférieures [S, résumé] ([arXiv:1411.4691](https://arxiv.org/abs/1411.4691)).
- **Haris, Witten & Simon (2016), « Convex modeling of interactions with strong heredity », JCGS 25(4):981-1004, DOI 10.1080/10618600.2015.1067217 (FAMILY).** Formulation convexe résolue par ADMM qui impose l'hérédité forte ; estimateur sans biais des degrés de liberté [S, résumé] ([arXiv:1410.3517](https://arxiv.org/abs/1410.3517)).
- **Yuan, Joseph & Zou (2009), « Structured variable selection and estimation », Ann. Appl. Stat. 3(4), DOI 10.1214/09-AOAS254**, et **Choi, Li & Zhu (2010), « Variable selection with the strong heredity constraint and its oracle property »** : garrote non négatif avec contraintes d'hérédité, et reparamétrisation γ_jk = δ_jk β_j β_k qui impose l'hérédité forte [S] ([ar5iv 1011.0610](https://ar5iv.labs.arxiv.org/html/1011.0610)).
- **Zhao & Leng (2016), « An analysis of penalized interaction models », Bernoulli 22(3):1937-1961, DOI 10.3150/15-BEJ715.** Analyse unifiée sous condition de valeur propre restreinte : erreur ℓ1 en s·√(log p₁/n) pour s coefficients non nuls. Les interactions ne sont ni gaussiennes ni sous-gaussiennes même si les effets principaux le sont, ce qui invalide les preuves usuelles [V, résumé] ([arXiv:1603.09138](https://arxiv.org/abs/1603.09138)).
- **Critiques de l'hérédité par les méthodes concurrentes.** Fan, Kong, Li & Lv écrivent que l'hérédité « can also be easily violated in some situations » et que les méthodes qui l'imposent peuvent « become inefficient or even fail » en ultra-haute dimension ; dans leurs simulations sans hérédité (modèles M3, M4), la méthode en deux étapes fondée sur SIS (« SIS2 ») « failed to detect interactions » [V, texte ar5iv] ([ar5iv 1605.08933](https://ar5iv.labs.arxiv.org/html/1605.08933)). Yu, Bien & Tibshirani jugent que la hiérarchie « might be too limiting in some problems » et que les méthodes en plusieurs étapes exigent que tous les effets principaux actifs soient détectés tôt [V, texte ar5iv] ([ar5iv 1907.08414](https://ar5iv.labs.arxiv.org/html/1907.08414)).

### Inferences

- Le critère (a) est une méthode en deux étapes sous hérédité forte, avec la norme des poids ridge à la place d'une sélection lasso. Sa justification théorique (Hao & Zhang) demande que les effets principaux importants soient lisibles dans le modèle linéaire et que les interactions actives soient peu nombreuses. Aucune des deux conditions n'est plausible avec des variables fortement corrélées dont les poids ridge sont répartis entre variables redondantes.
- L'hérédité n'est invariante que par translation et changement d'échelle de chaque variable, pas par rotation. Les états d'un réseau récurrent aléatoire n'ont pas de base privilégiée : une forme quadratique xᵀ B x de rang faible, exprimée dans une base aléatoire, donne une matrice B dense où l'hérédité n'a aucune raison d'être vérifiée coordonnée par coordonnée. L'égalité entre hérédité (a) et anti-hérédité (b) est le résultat attendu dans ce cas.
- Le résultat de Ye & Yang donne un appui théorique à l'observation que l'hérédité n'apporte pas de gain systématique, même dans le cadre creux.

### Gaps

- Les textes de Nelder (1977, « A reformulation of linear models », JRSS A 140:48-77), Hamada & Wu (1992, J. Quality Technology 24:130-137), Chipman (1996, « Bayesian variable selection with related predictors », Canadian J. Statistics 24:17-36) et Choi, Li & Zhu (2010, JASA 105:354-364) n'ont pas été ouverts ; leurs références bibliographiques complètes sont données de mémoire.
- Le texte intégral de hierNet n'a pas pu être lu (conversion ar5iv en erreur). Les arguments attribués à Cox et à McCullagh & Nelder (grands effets principaux plus susceptibles de produire des interactions) et le plan de simulation exact de hierNet ne sont pas vérifiés.
- Références de publication non vérifiées, données de mémoire : RAMP dans JASA 113(522):615-625 (2018) ; glinternet dans JCGS 24(3):627-654 (2015) ; la note de Hao & Zhang dans The American Statistician 71(4):291-297 (2017) ; Ye & Yang dans IEEE Trans. Information Theory (2019).
- Aucun article sur l'hérédité ne traite explicitement un régime dense ou diffus des interactions.

---

## 2. Criblage d'interactions sans hérédité, et le critère (c) (corrélation avec le résidu du modèle linéaire)

### Takeaway

Le critère (c) est, pour une sortie, exactement la statistique de criblage de *sprinter* (Yu, Bien & Tibshirani, « Reluctant interaction modeling ») : le carré de sd(r)·|cor(Z_ℓ, r)| vaut Cov(Z_ℓ, r)² / Var(Z_ℓ). Ses garanties supposent un petit ensemble cible d'interactions à signal fort, séparé des autres par un écart ; elles sont vides quand le signal est réparti sur des dizaines de milliers de produits.

### Cited Findings

- **Yu, Bien & Tibshirani, « Reluctant interaction modeling », arXiv:1907.08414 (sprinter).** Étape 1 : lasso de y sur les effets principaux X, résidu r = y − Xθ̂. Étape 2 : garder les interactions telles que sd(r)·|cor(Z_ℓ, r)| > η, ou les m plus grandes |cor(Z_ℓ, r)| avec m = ⌈n / log n⌉ conseillé ; les p² interactions sont parcourues une seule fois sans être stockées. Étape 3 : lasso de r sur tous les effets principaux et les interactions retenues [V, texte ar5iv] ([ar5iv 1907.08414](https://ar5iv.labs.arxiv.org/html/1907.08414)).
- **Principe de « réticence ».** « One should prefer main effects over interactions if all else is equal » ; le principe ne lie pas une interaction à ses effets principaux et « sprinter does not assume any structure among interactions » [V] ([ar5iv 1907.08414](https://ar5iv.labs.arxiv.org/html/1907.08414)).
- **Hypothèses de sprinter.** X centré sous-gaussien de covariance Σ, bruit sous-gaussien indépendant ; κ log p ≤ √n, condition plus stricte que pour le criblage d'effets principaux parce que les interactions ont des queues plus lourdes ; Cov(X, Z) = Φ peut être non nulle et l'« interaction pure » est W = Z − Φᵀ Σ⁻¹ X ; une condition de type « β-min » η(α) ≥ η* porte sur l'ensemble cible I(α). L'exemple gaussien détaillé contient une seule interaction vraie. La garantie du criblage « top-m » (annexe F, théorème 14) exige m ≥ k et un écart d'au moins η* entre la plus petite covariance normalisée des k interactions cibles et la plus grande des autres [V, texte ar5iv, deux lectures] ([ar5iv 1907.08414](https://ar5iv.labs.arxiv.org/html/1907.08414)).
- **Limites reconnues de sprinter.** « The success of sprinter hinges on capturing a small set » d'interactions ; l'étape 1 est un ajustement mal spécifié, donc le signal d'interaction pur dégrade les vitesses ; une interaction corrélée aux effets principaux peut être écartée, ce qui est voulu [V] ([ar5iv 1907.08414](https://ar5iv.labs.arxiv.org/html/1907.08414)).
- **Résultats empiriques de sprinter.** Sur Tripadvisor, RMSE 1,153 pour le lasso d'effets principaux, 1,125 pour le lasso hiérarchique en deux étapes, 1,091 pour sprinter. En simulation, le lasso sur toutes les paires (APL) ne gagne que si le signal est fort et concentré dans des interactions pures. L'article ne contient aucune comparaison avec un tirage aléatoire d'interactions [V] ([ar5iv 1907.08414](https://ar5iv.labs.arxiv.org/html/1907.08414)).
- **Fan, Kong, Li & Lv, « Interaction pursuit with feature screening and selection », arXiv:1605.08933 (IP).** Le criblage porte sur les *variables d'interaction* : on classe corr(X_k², Y²), on garde les variables au-dessus d'un seuil, puis on forme toutes les paires des variables retenues, ensemble qui « generally provides an overestimate ». Les effets principaux sont criblés à part par SIS. Hypothèses : modèle creux, queues sous-exponentielles, signal marginal minimal d'ordre n^(−κ), plus petite valeur propre de la covariance des variables augmentées bornée inférieurement. Limite reconnue : IP « can suffer from the same potential issues as the SIS », par exemple des variables de bruit très corrélées qui masquent des signaux faibles [V, texte ar5iv] ([ar5iv 1605.08933](https://ar5iv.labs.arxiv.org/html/1605.08933)).
- **Kong, Li, Fan & Lv, « Interaction pursuit in high-dimensional multi-response regression via distance correlation », Annals of Statistics, arXiv:1605.03315 (IPDC).** Criblage par corrélation de distance sur des variables transformées, pour réponses multiples dont le nombre peut diverger ; méthode « generally applicable beyond the heredity assumption » ; sure screening et inégalités oracle [V, résumé] ([arXiv:1605.03315](https://arxiv.org/abs/1605.03315)).
- **Fan, Kong, Li & Zheng (2015), « Innovated interaction screening for high-dimensional nonlinear classification », Annals of Statistics 43(3):1243-1272, DOI 10.1214/14-AOS1308 (IIS-SQDA).** Les variables sont transformées, puis seules p variables sont examinées au lieu des p(p+1)/2 interactions ; suit une analyse discriminante quadratique creuse. Sure screening avec p exponentiel en n [V, résumé] ([arXiv:1501.01029](https://arxiv.org/abs/1501.01029)).
- **Tang, Fang & Dong (2020), JMLR 21(19).** La matrice hessienne principale est estimée par un M-estimateur pénalisé (ADMM) ; les interactions sont lues sur ses entrées non nulles ; la méthode « requires no structural assumption on the hierarchy » [S, résumé] ([JMLR](https://www.jmlr.org/papers/v21/19-071.html)).
- **Wang, Jiang & Zhu (2021), « Penalized interaction estimation for ultrahigh dimensional quadratic regression », Statistica Sinica 31(3):1549-1570.** Effets principaux et interactions sont estimés séparément ; la matrice d'interaction minimise une perte convexe pénalisée ; l'article affirme « does not require the widely used heredity assumption » et respecte le principe d'invariance [V, résumé] ([arXiv:1901.07147](https://arxiv.org/abs/1901.07147)).
- **Thanei, Meinshausen & Shah (2018), « The xyz algorithm for fast interaction search in high-dimensional data », JMLR 19(37):1-42.** Des projections aléatoires ramènent la recherche d'interactions à un problème de paire la plus proche ; les interactions *fortes* sont trouvées en temps presque linéaire, les plus faibles en O(p^α), 1 < α < 2 [V, résumé] ([arXiv:1610.05108](https://arxiv.org/abs/1610.05108)).
- **Shah (2016), « Modelling interactions in high-dimensional data with backtracking », JMLR 17(207):1-31.** Des ensembles croissants d'interactions candidates sont construits itérativement à partir des ajustements précédents ; l'article rappelle que le criblage par effets principaux seuls manque les variables qui ne comptent que par interaction [V, résumé] ([arXiv:1208.1174](https://arxiv.org/abs/1208.1174)).
- **Sure independence screening (Fan & Lv 2008, JRSS-B).** SIS classe les variables par corrélation marginale. Il échoue quand une variable est importante conjointement mais non corrélée marginalement à la réponse, ou quand des variables non pertinentes ont des corrélations marginales plus fortes ; la version itérative applique SIS au résidu de la régression sur les variables déjà retenues [S] ([Zhang, Jiang & Lan, arXiv:1812.01367](https://arxiv.org/abs/1812.01367) ; [Fan & Lv, revue, arXiv:0910.1122](https://arxiv.org/abs/0910.1122)).
- **Li, Kong, Fan & Lv (2022), JBES.** Sélection d'interactions en régression quadratique ultra-haute dimension, avec les mêmes inégalités oracle que le lasso et une borne explicite sur le taux de faux signes [S] ([EconPapers](https://econpapers.repec.org/RePEc:taf:jnlbes:v:40:y:2022:i:3:p:1234-1245)).

### Inferences

- Écarts entre le critère (c) et sprinter : le modèle de l'étape 1 est un ridge et non un lasso ; la somme porte sur 29 sorties ; le nombre de produits gardés (quelques milliers) dépasse largement m = n / log n ; le modèle final est un ridge sur les produits retenus. La statistique de classement reste la même.
- La garantie « top-m » de sprinter demande un écart entre les interactions cibles et les autres. Avec une importance répartie sur des dizaines de milliers de produits, cet écart n'existe pas et le classement marginal n'a pas de garantie.
- Le criblage marginal note chaque produit isolément et ne tient pas compte de la redondance entre produits retenus. Avec des variables corrélées, les produits les mieux notés sont fortement corrélés entre eux et couvrent un sous-espace étroit, alors qu'un tirage uniforme couvre mieux l'espace des produits. Ce mécanisme explique qu'un criblage puisse être *pire* que le hasard à nombre de produits égal ; il n'a pas été trouvé énoncé tel quel dans une source.
- IP et la « marginal epistasis » (section 3) notent des *variables* par leur implication totale dans des interactions, et non des paires. Ce type de score par variable, suivi de tous les produits entre variables retenues, est une variante non testée du critère (a).

### Gaps

- Les prénoms des auteurs de sprinter (Guo Yu, Jacob Bien, Ryan Tibshirani) et une éventuelle publication en revue ne sont pas vérifiés.
- Niu, Hao & Zhang (2018), « Interaction screening by partial correlation », Statistics and Its Interface : seul un résumé de moteur de recherche a été vu (criblage par corrélation partielle, compatible avec Pearson, Spearman et Kendall). Hypothèses et garanties non vérifiées.
- Shah & Meinshausen (2014), « Random intersection trees », JMLR 15:629-654 : non ouvert.
- Le détail de la transformation « innovée » de IIS (multiplication par la matrice de précision, de mémoire) n'est pas vérifié dans le texte.
- La publication en revue de IP (arXiv:1605.08933) n'est pas confirmée ; celle de IPDC (Annals of Statistics 45(2):897-922, 2017) est donnée de mémoire.
- L'article original de Fan & Lv (2008, JRSS-B 70:849-911) n'a pas été ouvert.

---

## 3. Échecs connus : variables corrélées, signal diffus ou dense, ridge contre lasso

### Takeaway

Aucun résultat publié n'a été trouvé qui énonce directement que l'hérédité ou le criblage par le résidu ne bat pas le hasard quand les interactions sont denses. Les éléments disponibles sont indirects : le criblage marginal échoue avec des variables corrélées, la covariance marginale d'un produit avec le résidu mesure Σ H̄ Σ et non H̄, la parcimonie est une hypothèse liée à la base, et la littérature de génétique quantitative traite le cas dense par un ridge sur toutes les paires.

### Cited Findings

- **Hessienne principale (Li 1992, « On principal Hessian directions for data visualization and dimension reduction: another application of Stein's lemma », JASA 87(420):1025-1039).** Pour x gaussien, le lemme de Stein donne H̄ = Σ⁻¹ Σ_yxx Σ⁻¹, avec Σ_yxx = E[(y − ȳ)(x − μ)(x − μ)ᵀ]. Li propose une variante fondée sur le résidu r de la régression linéaire, car ajouter une fonction linéaire de x à y ne change pas H̄ [S] ([tutoriel SDR, arXiv:2110.09620](https://arxiv.org/abs/2110.09620) ; [DOI de l'article](https://doi.org/10.1080/01621459.1992.10476258)).
- La version par résidu de pHd a une variance asymptotique plus faible que la version par réponse centrée en régression quadratique [S, résumé] ([Statistics & Probability Letters 126:108-113, 2017](https://econpapers.repec.org/RePEc:eee:stapro:v:126:y:2017:i:c:p:108-113)).
- **Échec de SIS avec des variables corrélées** : voir la section 2 ([arXiv:1812.01367](https://arxiv.org/abs/1812.01367)), et la limite héritée par IP ([ar5iv 1605.08933](https://ar5iv.labs.arxiv.org/html/1605.08933)).
- **Dépendance à la paramétrisation** : l'hérédité dépend du codage et la justification des méthodes en deux étapes demande Cov(X, Z) bloc-diagonale [V] ([ar5iv 1412.7138](https://ar5iv.labs.arxiv.org/html/1412.7138)).
- **Invariance par rotation (Ng 2004, « Feature selection, L1 vs. L2 regularization, and rotational invariance », ICML, DOI 10.1145/1015330.1015435).** La régression logistique L1 a une complexité en échantillons logarithmique en le nombre de variables non pertinentes ; les algorithmes invariants par rotation, dont la régression L2, ont une complexité au pire au moins linéaire [S, résumé] ([ICML 2004](https://icml.cc/Conferences/2004/proceedings/abstracts/354.htm)).
- **« Bet on sparsity » (Hastie, Tibshirani & Friedman).** « Use a procedure that does well in sparse problems, since no procedure does well in dense problems ». Tibshirani précise que les méthodes ℓ1 supposent la vérité creuse « in some basis » ; quand le nombre de coefficients actifs est grand, ridge fait mieux que le lasso [S] ([discussion de Gelman et Tibshirani, 2013](https://statmodeling.stat.columbia.edu/2013/12/16/whither-the-bet-on-sparsity-principle-in-a-nonsparse-world/)).
- **Giannone, Lenza & Primiceri (2021), « Economic predictions with big data: the illusion of sparsity », Econometrica 89(5):2409-2437, DOI 10.3982/ECTA17842.** Avec un a priori qui sépare sélection et rétrécissement, la loi a posteriori ne se concentre pas sur un modèle creux mais sur un large ensemble de modèles avec beaucoup de prédicteurs ; la parcimonie n'apparaît que sous un fort a priori en sa faveur [S, résumé de l'éditeur] ([Econometrica](https://econometricsociety.org/doi/10.3982/ECTA17842)).
- **Ridge sur toutes les paires en forme duale (HiQR : Wang, Chen & Jiang, Computational Statistics & Data Analysis, arXiv:2306.00469).** La régression quadratique ridge a une forme close qui ne demande que le système n × n construit sur λ I_n + n⁻¹ (XXᵀ)∘(XXᵀ), au coût O(n p² + n³) au lieu de O(p⁶). L'article ne suppose ni hérédité ni loi particulière, mais ne discute pas le cas d'une matrice d'interaction dense [V, texte ar5iv] ([ar5iv 2306.00469](https://ar5iv.labs.arxiv.org/html/2306.00469)).
- **Génétique quantitative : épistasie dense.** Le modèle EG-BLUP met un effet aléatoire gaussien (ridge) sur toutes les interactions par paires de marqueurs, avec une matrice de parenté égale au produit de Hadamard des matrices additives (Jiang & Reif 2015, Genetics 201:759-768 ; Martini et al. 2016, Theor. Appl. Genet. 129:963-976) [V pour la description, dans Vojgani et al.] ([PMC8354961](https://pmc.ncbi.nlm.nih.gov/articles/PMC8354961)).
- **Vojgani et al. (2021), Theor. Appl. Genet. 134(9):2913-2930, DOI 10.1007/s00122-021-03868-1.** Le modèle avec *toutes* les interactions (ERRBLUP) donne une capacité prédictive « almost identical » au modèle additif. Garder les interactions de plus grande variance d'effet *estimée dans le modèle complet* (sERRBLUP) améliore la prédiction, au mieux avec 1 % ou 0,1 % des interactions ; une sélection trop sévère passe parfois sous le modèle additif. Le gain vient surtout d'une sélection faite sur les données d'un *autre environnement* ; dans le même environnement le gain est marginal. Les auteurs reconnaissent un gain « somewhat inflated » parce que le meilleur de plusieurs modèles est comparé à un seul [V, texte PMC] ([PMC8354961](https://pmc.ncbi.nlm.nih.gov/articles/PMC8354961)).
- **Crawford, Zeng, Mukherjee & Zhou (2017), PLoS Genetics, DOI 10.1371/journal.pgen.1006869 (MAPIT).** Le test d'épistasie marginale évalue pour chaque variant la somme de ses interactions avec tous les autres, par un modèle à composantes de variance, au lieu de chercher les paires une à une [S, résumé] ([PMC5550000](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5550000/)).

### Inferences

- La matrice des covariances marginales du critère (c), C_ij = Cov(x_i x_j, r), est la matrice Σ_rxx de la pHd par résidu. Pour x gaussien, C = Σ H̄ Σ. Le criblage élément par élément classe donc les entrées de Σ H̄ Σ et non celles de H̄. Avec des variables très corrélées (Σ de rang effectif faible), C est dominée par les directions principales de Σ, et les produits les mieux notés sont ceux des variables chargées sur ces directions, quelle que soit la structure de H̄. C'est une explication directe de l'échec du critère (c), déduite de la formule de Li et non lue dans une source.
- Les méthodes qui corrigent cet effet travaillent sur Σ⁻¹ C Σ⁻¹ (pHd, Tang-Fang-Dong, Wang-Jiang-Zhu) ou transforment les variables avant criblage (IIS). Elles supposent toutes H̄ creuse.
- Le noyau quadratique (xᵀx′)² et le ridge sont invariants par rotation de x, alors que l'hérédité, le criblage et le lasso ne le sont pas. Si les variables sont des projections aléatoires d'un signal de faible dimension, aucun de ces critères liés à la base n'a de raison d'isoler de bons produits.
- La littérature de génétique indique un critère supervisé qui fonctionne parfois dans un régime dense : classer les interactions par l'amplitude de leur coefficient dans le *modèle ridge complet*, avec une sélection validée sur des données indépendantes. Ce critère diffère de (a) et de (c).

### Gaps

- Aucun théorème ni étude empirique trouvés qui comparent explicitement hérédité, criblage par résidu et tirage aléatoire dans un régime d'interactions denses ou de rang faible.
- Aucune source trouvée sur la sélection d'interactions pour des variables qui sont des projections aléatoires d'un signal de faible dimension.
- L'article de Li (1992) n'a pas été ouvert ; la formule H̄ = Σ⁻¹ Σ_yxx Σ⁻¹ vient de sources secondaires. Le titre et les auteurs exacts de l'article de 2017 (Cheng et Zhu d'après le résumé de recherche) ne sont pas vérifiés.
- Pistes repérées mais non ouvertes : « Bad estimation, good prediction: the Lasso in dense regimes » (arXiv:2502.07959) ; Dobriban & Wager (2018) sur ridge en régime dense ; Hall & Xue (2014) ; les résumés de Martini et al. (2016) et de Jiang & Reif (2015), connus seulement par Vojgani et al.
- Le passage exact du livre de Hastie, Tibshirani & Friedman (section 16.2.2, de mémoire) n'a pas été vérifié.

---

## 4. Méthodes conçues pour une structure d'interaction dense ou de rang faible

### Takeaway

Quand la partie quadratique est xᵀ B x avec B dense mais de rang faible, la littérature paramètre B par ses facteurs (machines de factorisation, réseaux polynomiaux, pénalité nucléaire) ou cherche d'abord les directions utiles (pHd, SAVE, modèles multi-indices), au lieu de sélectionner des entrées de B.

### Cited Findings

- **Rendle (2010), « Factorization machines », ICDM, p. 995-1000, DOI 10.1109/ICDM.2010.127.** Le poids de l'interaction entre les variables i et j est le produit scalaire ⟨v_i, v_j⟩ de vecteurs latents de dimension k : la matrice d'interaction est de rang au plus k. Le modèle est conçu pour des données creuses [S] ([DOI](https://doi.org/10.1109/ICDM.2010.127)).
- **Blondel, Ishihata, Fujino & Ueda (2016), « Polynomial networks and factorization machines: new insights and efficient training algorithms », ICML.** Les deux modèles sont vus comme l'estimation d'un tenseur symétrique de rang faible, résolue par optimisation multi-convexe [V, résumé] ([arXiv:1607.08810](https://arxiv.org/abs/1607.08810)).
- **Pénalité nucléaire sur B.** HiQR couvre la norme nucléaire ‖B‖∗ et l'hybride ℓ1 + nucléaire (creux plus rang faible) par ADMM [V] ([ar5iv 2306.00469](https://ar5iv.labs.arxiv.org/html/2306.00469)). Lu, Zhu & Lian (2023), « Sparse and low-rank matrix quantile estimation with application to quadratic regression », Statistica Sinica 33:945-959, supposent la matrice de coefficients de rang faible et/ou creuse [S] ([Statistica Sinica](https://www3.stat.sinica.edu.tw/statistica/J33N2/j33n216/j33n216.html)).
- **Hung & Wang (2013), « Matrix variate logistic regression model with application to EEG data », Biostatistics 14(1):189-202.** La matrice de paramètres est approchée au rang 1 [S] ([arXiv:1105.2150](https://arxiv.org/abs/1105.2150)).
- **Directions hessiennes principales (Li 1992).** Les vecteurs propres de Σ⁻¹ Σ_yxx Σ⁻¹ associés aux valeurs propres non nulles appartiennent au sous-espace central ; la méthode n'est pas exhaustive quand la hessienne moyenne est singulière [S] ([tutoriel SDR, arXiv:2110.09620](https://arxiv.org/abs/2110.09620)). Une revue récente couvre les algorithmes pour modèles multi-indices [S, titre seul] ([arXiv:2504.05426](https://arxiv.org/abs/2504.05426)).
- **PLS non linéaire (Wold, Kettaneh-Wold & Skagerberg 1989, « Nonlinear PLS modeling », Chemometrics and Intelligent Laboratory Systems 7:53-65).** La relation interne entre le score latent des prédicteurs et celui de la réponse est un polynôme du second degré, ajusté séparément pour chaque paire de scores [S] ([revue du PLS non linéaire](https://www.igi-global.com/chapter/nonlinear-partial-least-squares-overview/45470/)).
- **GMDH (Ivakhnenko).** Chaque nœud combine une paire d'entrées par un polynôme du second degré a + b x_i + c x_j + d x_i² + e x_j² + f x_i x_j ; les nœuds sont empilés en couches avec sélection [S] ([vignette GMDHreg](https://search.r-project.org/CRAN/packages/GMDHreg/vignettes/GMDHreg.html)).
- **Analyse discriminante quadratique creuse** (IIS-SQDA) : voir la section 2 ([arXiv:1501.01029](https://arxiv.org/abs/1501.01029)).

### Inferences

- Un modèle complet avec 2 000 à 3 000 paramètres effectifs sur 129 000 à 500 000 produits est compatible avec une matrice B de rang faible par sortie. Les machines de factorisation et la pénalité nucléaire sont les paramétrisations directes de cette hypothèse.
- Pour 29 sorties, les B_k forment un tenseur p × p × 29 ; un sous-espace commun aux B_k (modèle multi-indices partagé) réduirait le problème à tous les produits de d directions apprises, soit d(d+1)/2 termes. C'est la version supervisée du critère (f).

### Gaps

- Aucune référence « Wang, Zhou et al., low-rank quadratic » identifiée ; la référence la plus proche trouvée est Lu, Zhu & Lian (2023).
- Non ouverts, cités de mémoire : Livni, Shalev-Shwartz & Shamir (2014, NIPS, réseaux polynomiaux) ; Blondel, Fujino, Ueda & Ishihata (2016, NIPS, « Higher-order factorization machines ») ; Cook & Weisberg (1991, SAVE) ; Zhou & Li (2014, « Regularized matrix regression », JRSS-B) ; Friedman & Stuetzle (1981, projection pursuit regression).
- Aucune méthode nommée de « régression sur composantes principales avec interactions entre composantes » n'a été trouvée ; la recherche ne renvoie que le PLS quadratique.
- L'article de Rendle n'a pas été lu directement ; l'article original d'Ivakhnenko non plus.

---

## 5. Contrepartie nommée de chacun de nos critères

### Takeaway

Seuls (a) et (c) ont une contrepartie exacte publiée (sélection en deux étapes sous hérédité forte ; étape 2 de sprinter). Les critères (d), (e), (f) relèvent de la réduction de dimension non supervisée d'une régression ridge (sous-échantillonnage de colonnes, projection aléatoire, PCR), comparés entre eux par Slawski (2018).

### Cited Findings

| Critère | Contrepartie la plus proche | Statut |
|---|---|---|
| (a) produits entre les m variables de plus grand poids linéaire | Méthodes en deux étapes sous hérédité forte : lasso en deux étapes (Hao, Feng & Zhang), « SIS2 » (référence dans IP), « two-stage hierarchical lasso » (référence dans sprinter), iFORM | Contrepartie exacte à la règle de sélection près |
| (b) idem avec les plus petits poids | Scénario « anti-hiérarchique » des simulations de glinternet | Contrôle, pas une méthode |
| (c) Σ_k Cov(x_i x_j, r_k)² / Var(x_i x_j) | Étape 2 de sprinter ; SIS itératif appliqué aux produits ; entrées de la matrice Σ_rxx de la pHd par résidu | Contrepartie exacte pour une sortie |
| (d) paires tirées uniformément | Sous-échantillonnage uniforme des colonnes de la matrice des produits (Slawski) | Contrepartie exacte, sans nom propre aux interactions |
| (e) produits entre m variables aléatoires | Sous-échantillonnage de colonnes de x, puis expansion quadratique complète | Pas de nom trouvé |
| (f) produits des m premières composantes principales | PCR suivie d'une expansion quadratique ; noyau quadratique sur les entrées réduites par ACP | Pas de nom trouvé |
| Produits des sorties du modèle linéaire | Modèle multi-indices à directions fixées ; PLS quadratique ; GMDH ; empilement (stacking) | Pas de contrepartie exacte trouvée |

- Sources des lignes (a), (b), (c) : [arXiv:1501.00049](https://arxiv.org/abs/1501.00049) [V], [ar5iv 1605.08933](https://ar5iv.labs.arxiv.org/html/1605.08933) [V], [ar5iv 1907.08414](https://ar5iv.labs.arxiv.org/html/1907.08414) [V], [ar5iv 1308.2719](https://ar5iv.labs.arxiv.org/html/1308.2719) [V].
- **Slawski (2018), « On principal components regression, random projections, and column subsampling », Electronic Journal of Statistics 12(2):3673-3712, DOI 10.1214/18-EJS1486.** En plan fixe, avec un rang de X du même ordre que n, les projections aléatoires de type Johnson-Lindenstrauss atteignent une erreur de prédiction proche de celle de la PCR avec k = Ω(r log n) projections pour un paramètre de rang r. Pour le sous-échantillonnage de colonnes : « We leave it as an open problem whether a similar result can be established », et « more stringent conditions are required ». La proposition 3 donne l'équivalence en risque attendu avec les projections gaussiennes quand X a un spectre plat (n ≥ d) ou des entrées gaussiennes i.i.d. Le sous-échantillonnage de colonnes se dégrade fortement pour des entrées de Cauchy, donne souvent une erreur de test moyenne plus faible sur données réelles pour k petit mais avec plus de variabilité, et « the relative performance will depend on properties of » X. La dimension réduite k règle le compromis biais-variance, le biais étant contrôlé par la queue du spectre au-delà du rang r [V, texte ar5iv] ([ar5iv 1709.08104](https://ar5iv.arxiv.org/html/1709.08104)).
- **Variables aléatoires pour noyaux polynomiaux.** Kar & Karnick (2012), « Random feature maps for dot product kernels », AISTATS, PMLR 22:583-591 : cartes « Random Maclaurin » avec approximation uniforme du noyau sur un compact [S, résumé] ([PMLR](https://proceedings.mlr.press/v22/kar12.html)). Hamid et al., « Compact random feature maps » (CRAFTMaps), arXiv:1312.4626 : la construction précédente peut être déficiente en rang pour les noyaux polynomiaux [S] ([arXiv:1312.4626](https://arxiv.org/abs/1312.4626)). Ahle et al., « Oblivious sketching of high-degree polynomial kernels », arXiv:1909.01410 : esquisse indépendante des données d'un produit tensoriel sans le former [V, résumé] ([arXiv:1909.01410](https://arxiv.org/abs/1909.01410)).
- PLS quadratique et GMDH : voir la section 4.

### Inferences

- Le critère (d) est un sous-échantillonnage de coordonnées de la carte x ↦ vech(x xᵀ) du noyau quadratique. Il diffère de Random Maclaurin, dont chaque variable est un produit (ωᵀx)(ω′ᵀx) de deux projections aléatoires, et de TensorSketch.
- La condition de Slawski pour que le sous-échantillonnage de colonnes vaille une projection aléatoire (spectre plat, entrées de type gaussien, pas de queues lourdes) correspond à une matrice de produits dont l'information est répartie uniformément entre colonnes. Des états de réseau récurrent aléatoire, denses et mélangés, sont dans ce cas favorable.
- La seconde étape quadratique sur les 29 sorties du modèle linéaire (435 produits) est un modèle multi-indices dont les directions sont les lignes de la matrice de poids linéaire. Elle suppose que la partie quadratique vit dans le sous-espace de dimension 29 déjà utilisé par la partie linéaire, hypothèse que pHd par résidu permet de tester (la pHd par résidu cherche justement les directions de courbure, qui peuvent différer des directions linéaires).

### Gaps

- Aucune source trouvée pour « classifieur quadratique sur les scores LDA » ou « expansion polynomiale des composantes PLS/LDA » en tant que méthode nommée. Non vérifiés, de mémoire : Wolpert (1992, « Stacked generalization »), Hastie, Tibshirani & Buja (1994, « Flexible discriminant analysis by optimal scoring »), Ho (1998, méthode des sous-espaces aléatoires).
- Pham & Pagh (2013, TensorSketch, KDD) et Avron, Nguyen & Woodruff (2014) non ouverts.
- La dépendance de la dimension d'esquisse à la dimension statistique dans Ahle et al. n'apparaît pas dans le résumé et n'est pas vérifiée.

---

## 6. Énoncés selon lesquels des sous-ensembles aléatoires sont compétitifs

### Takeaway

Aucun énoncé publié propre aux termes d'interaction n'a été trouvé. Les appuis sont dans la théorie des approximations aléatoires de la régression ridge à noyau : le nombre de variables ou de colonnes nécessaire est de l'ordre des degrés de liberté effectifs du problème, et un tirage indépendant des données suffit quand l'information est répartie uniformément. Ce cadre prédit l'observation que la précision dépend surtout du *nombre* de produits.

### Cited Findings

- **Bach (2013), « Sharp analysis of low-rank kernel matrix approximations », COLT.** Pour la régression ridge à noyau avec une approximation construite sur un sous-ensemble aléatoire de colonnes de la matrice de noyau, le rang p peut être choisi « linear in the degrees of freedom associated with the problem », quantité vue comme le nombre implicite de paramètres de l'estimateur [V, résumé] ([arXiv:1208.2015](https://arxiv.org/abs/1208.2015)).
- **Rudi & Rosasco (2017), « Generalization properties of learning with random features », NIPS.** Le ridge sur variables aléatoires atteint des bornes d'apprentissage en O(1/√n) avec environ O(√n log n) variables ; les vitesses plus rapides demandent davantage de variables, sauf si elles sont tirées selon une loi « possibly problem dependent » [V, résumé] ([arXiv:1602.04474](https://arxiv.org/abs/1602.04474)).
- **Lu, Dhillon, Foster & Ungar (2013), « Faster ridge regression via the subsampled randomized Hadamard transform », NIPS, p. 369-377.** Pour p ≫ n, un préconditionnement de Hadamard aléatoire suivi d'un sous-échantillonnage de *variables* donne un ridge en O(n p log n), avec des bornes de risque en plan fixe [S, résumé] ([NeurIPS](https://proceedings.neurips.cc/paper/2013/hash/621bf66ddb7c962aa0d22ac97d69b793-Abstract.html)).
- **Slawski (2018)** : conditions et limites du sous-échantillonnage uniforme de colonnes face à la PCR, voir la section 5 ([ar5iv 1709.08104](https://ar5iv.arxiv.org/html/1709.08104)).
- **Rajabinasab, Houle, Chelly & Zimek (2026), « Worse than random: the importance of a baseline for unsupervised feature selection », arXiv:2605.22973.** Beaucoup de méthodes récentes de sélection *non supervisée* de variables font moins bien qu'un tirage aléatoire, en qualité et en coût ; les auteurs demandent que le tirage aléatoire devienne une référence obligatoire [V, résumé] ([arXiv:2605.22973](https://arxiv.org/abs/2605.22973)).
- **Éléments contraires.** Vojgani et al. obtiennent un gain avec une sélection supervisée des interactions par variance d'effet dans le modèle complet, validée sur un autre environnement (section 3) ([PMC8354961](https://pmc.ncbi.nlm.nih.gov/articles/PMC8354961)). AutoFIS (Liu et al., KDD 2020) retire 50 à 80 % des interactions d'ordre 2 d'une machine de factorisation avec un gain d'AUC ; aucune comparaison avec un retrait aléatoire n'a été confirmée [S] ([arXiv:2003.11235](https://arxiv.org/abs/2003.11235)).
- xyz et les « random intersection trees » utilisent l'aléa pour *trouver* des interactions fortes, pas pour en tirer un sous-ensemble arbitraire ([arXiv:1610.05108](https://arxiv.org/abs/1610.05108)).

### Inferences

- La lecture la plus directe du résultat négatif est celle de l'approximation de rang faible d'un ridge à noyau : la matrice des produits a un rang effectif d'environ 2 000 à 3 000, et tout ensemble de quelques milliers de colonnes qui engendre bien ce sous-espace donne une précision proche. Un tirage uniforme y parvient quand les scores de levier des colonnes sont à peu près uniformes. Un critère supervisé qui concentre les colonnes dans une région (variables de fort poids, produits corrélés au résidu) engendre un sous-espace plus étroit à nombre de colonnes égal.
- Dans ce cadre, les sélections qui devraient battre le hasard sont celles qui visent la couverture du sous-espace (scores de levier ridge des colonnes, sélection gloutonne de type pivot de Cholesky ou sélection orthogonale ascendante sur le résidu), et non un score marginal par produit. Cette prédiction vient de la littérature d'approximation de noyaux citée ici, pas d'une étude sur les interactions.
- Le résultat de Rajabinasab et al. porte sur des variables et non sur des produits, et sur la sélection non supervisée ; il appuie seulement la pratique de la référence aléatoire.

### Gaps

- Aucune étude trouvée qui compare, pour des termes d'interaction, un sous-ensemble aléatoire à un sous-ensemble sélectionné et conclut à l'égalité.
- Non ouverts : Avron et al. (2017) sur le lien entre nombre de variables de Fourier aléatoires et dimension statistique ; Bach (2017, JMLR) sur l'échantillonnage optimal par scores de levier ; Tropp (2011) et Mahoney (2011) sur le sous-échantillonnage uniforme après rotation et la cohérence ; Yang et al. (2012) sur Nyström contre variables aléatoires ; LeJeune, Javadi & Baraniuk (2020) sur l'équivalence entre ensembles à sous-échantillonnage de variables et ridge.
- Le texte intégral d'AutoFIS n'a pas été lu ; la présence d'une référence aléatoire reste à vérifier.
- La borne de risque exacte de Lu et al. (2013) n'a pas été lue.
