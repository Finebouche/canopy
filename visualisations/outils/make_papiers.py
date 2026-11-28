"""Writes canopy/visualisations/papiers.html: one visual card per planned paper (key figure, state, idea, angle,
missing pieces, references, venues). Run: python make_papiers.py"""
from html import escape

OUT = "/Users/tanguy/Code/Finebouche/canopy/visualisations/papiers.html"


def dots(rows, lo, hi, unit="%"):
    """A small dot plot: rows (label, value, highlighted, note), on a scale lo..hi."""
    w, left, right, rh = 320, 104, 46, 24
    h = len(rows) * rh + 22
    x = lambda v: left + (v - lo) / (hi - lo) * (w - left - right)
    out = [f'<svg class="fig" viewBox="0 0 {w} {h}" role="img" aria-label="'
           + escape(", ".join(f"{l} {str(v).replace('.', ',')} {unit}" for l, v, *_ in rows)) + '">']
    for t in range(int(lo), int(hi) + 1, 2 if hi - lo <= 12 else 5):
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="4" y2="{h - 16}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{h - 4}" class="tick" text-anchor="middle">{t}</text>')
    for i, (label, value, hot, *rest) in enumerate(rows):
        y = 14 + i * rh
        cls = "hot" if hot else "cold"
        out.append(f'<text x="{left - 8}" y="{y + 4}" class="lab {cls}" text-anchor="end">{escape(label)}</text>'
                   f'<line x1="{x(lo):.1f}" x2="{x(value):.1f}" y1="{y}" y2="{y}" class="stem {cls}"/>'
                   f'<circle cx="{x(value):.1f}" cy="{y}" r="5" class="dot {cls}"/>'
                   f'<text x="{x(value) + 9:.1f}" y="{y + 4}" class="val {cls}">{str(value).replace(".", ",")}'
                   + (f' <tspan class="tick">{escape(rest[0])}</tspan>' if rest and rest[0] else "") + '</text>')
    out.append("</svg>")
    return "".join(out)


def bars(rows, hi, unit="pt"):
    """Bars from zero: rows (label, value, highlighted, note)."""
    w, left, right, rh = 320, 104, 70, 26
    h = len(rows) * rh + 8
    x = lambda v: left + v / hi * (w - left - right)
    out = [f'<svg class="fig" viewBox="0 0 {w} {h}" role="img" aria-label="'
           + escape(", ".join(f"{l} +{str(v).replace('.', ',')} {unit}" for l, v, *_ in rows)) + '">']
    for i, (label, value, hot, note) in enumerate(rows):
        y = 6 + i * rh
        cls = "hot" if hot else "cold"
        out.append(f'<text x="{left - 8}" y="{y + 13}" class="lab {cls}" text-anchor="end">{escape(label)}</text>'
                   f'<rect x="{left}" y="{y + 3}" width="{max(x(value) - left, 2):.1f}" height="14" rx="3" class="bar {cls}"/>'
                   f'<text x="{x(value) + 6:.1f}" y="{y + 14}" class="val {cls}">+{str(value).replace(".", ",")} {unit}'
                   + (f' <tspan class="tick">{escape(note)}</tspan>' if note else "") + '</text>')
    out.append("</svg>")
    return "".join(out)


def flow(steps, loop=""):
    """A pipeline of boxes joined by arrows, the last ones highlighted."""
    w, n = 320, len(steps)
    bw, gap = (w - 4 - (n - 1) * 14) / n, 14
    out = [f'<svg class="fig" viewBox="0 0 {w} {96 if loop else 66}" role="img" aria-label="{escape(" puis ".join(s for s, _ in steps) + (", puis de nouveau : " + loop if loop else ""))}">',
           '<defs><marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">'
           '<path d="M0 0L8 4L0 8z" class="arrowhead"/></marker></defs>']
    for i, (label, hot) in enumerate(steps):
        x0 = 2 + i * (bw + gap)
        cls = "hot" if hot else "cold"
        words = label.split(" ")
        out.append(f'<rect x="{x0:.1f}" y="8" width="{bw:.1f}" height="50" rx="8" class="box {cls}"/>')
        for k, wd in enumerate(words):
            out.append(f'<text x="{x0 + bw / 2:.1f}" y="{33 - 6 * (len(words) - 1) + 12 * k}" class="boxt {cls}" text-anchor="middle">{escape(wd)}</text>')
        if i < n - 1:
            out.append(f'<line x1="{x0 + bw + 1:.1f}" x2="{x0 + bw + gap - 2:.1f}" y1="33" y2="33" class="arrow" marker-end="url(#ar)"/>')
    if loop:  # (the last box feeds the first one again: a return arrow under the boxes, and its label)
        xa, xb = 2 + (n - 1) * (bw + gap) + bw / 2, 2 + bw / 2
        out.append(f'<path d="M{xa:.1f} 59 V78 Q{xa:.1f} 84 {xa - 6:.1f} 84 H{xb + 6:.1f} Q{xb:.1f} 84 {xb:.1f} 78 V61" '
                   f'class="arrow loop" fill="none" marker-end="url(#ar)"/>'
                   f'<rect x="{(xa + xb) / 2 - 88:.1f}" y="77" width="176" height="14" class="looplab-bg"/>'
                   f'<text x="{(xa + xb) / 2:.1f}" y="88" class="looplab" text-anchor="middle">{escape(loop)}</text>')
    out.append("</svg>")
    return "".join(out)


def grid(rows, cols, filled):
    """A small matrix of what a review compares: rows × cols, filled cells done."""
    w, left, top, cw, ch = 320, 92, 22, 54, 20
    h = top + len(rows) * ch + 4
    out = [f'<svg class="fig" viewBox="0 0 {w} {h}" role="img" aria-label="Règles par ordre de lecture">']
    for j, c in enumerate(cols):
        out.append(f'<text x="{left + j * cw + cw / 2}" y="14" class="tick" text-anchor="middle">{escape(c)}</text>')
    for i, r in enumerate(rows):
        out.append(f'<text x="{left - 8}" y="{top + i * ch + 14}" class="lab cold" text-anchor="end">{escape(r)}</text>')
        for j, _ in enumerate(cols):
            cls = "hot" if (i, j) in filled else "cold"
            out.append(f'<rect x="{left + j * cw + 3}" y="{top + i * ch + 3}" width="{cw - 6}" height="{ch - 6}" rx="3" class="cell {cls}"/>')
    out.append("</svg>")
    return "".join(out)


# prestige of each venue, an estimate on three levels (3: very selective, generalist or top of its field; 2: a
# reference of its field; 1: specialized or accessible), not an impact factor
PRESTIGE = {"PLoS Comput. Biol.": 2, "eLife": 3, "Neural Networks": 2, "ICASSP": 2, "Bioacoustics": 1, "Current Biology": 3,
            "Proc. R. Soc. B": 2, "Animal Behaviour": 2, "ESANN": 1, "IJCNN": 1, "ICANN": 1, "Neurocomputing": 1, "Chaos": 2,
            "Neural Comput.": 2, "Phys. Rev. E": 2, "Sci. Rep.": 1, "Front. Comput. Neurosci.": 1, "NeurIPS": 3,
            "Physics Reports": 3, "ICML": 3, "IJCAI": 3, "Science": 3, "Nature": 3, "Nature Commun.": 3, "PNAS": 3, "COLT": 3,
            "ICPR": 2, "EUSIPCO": 2, "ECAI": 2, "PRICAI": 1, "ICONIP": 1, "ICASSP 2028": 2}
CONFERENCES = {"ICASSP", "ESANN", "IJCNN", "ICANN", "NeurIPS", "ICML", "IJCAI", "COLT",
               "ICPR", "EUSIPCO", "ECAI", "PRICAI", "ICONIP", "ICASSP 2028"}  # (the others are journals)
# (acceptance rates found on the web on 2026-11-25: NeurIPS and ICML 2025, ICASSP 2024, IJCNN and ICANN 2025)
ACCEPTED = {"NeurIPS": "25 %", "ICML": "27 %", "ICASSP": "48 %", "IJCNN": "39 %", "ICANN": "45 %"}
LEVELS = {3: "très sélectif", 2: "référence du domaine", 1: "spécialisé ou accessible"}


# links of the references (DOI or page of the paper, checked on the web on 2026-11-25) and of the venues (the journal,
# or the next edition of the conference when it is announced: NEXT, shown when the mouse is over the venue)
DOI = "https://doi.org/"
REF_URLS = {
    ("Markowitz et al.", 2013): DOI + "10.1371/journal.pcbi.1003052",
    ("Gardner, Naef & Nottebohm", 2005): DOI + "10.1126/science.1108214",
    ("Gardner et al.", 2005): DOI + "10.1126/science.1108214",
    ("Pagliarini et al.", 2021): "https://hal.science/hal-03244723",
    ("Trouvain & Hinaut", 2021): DOI + "10.1007/978-3-030-86383-8_6",
    ("Cohen et al.", 2022): DOI + "10.7554/eLife.63853",
    ("Roeske et al.", 2020): DOI + "10.1016/j.cub.2020.06.072",
    ("James, Sakata et al.", 2026): DOI + "10.1126/science.adr9983",
    ("Cohen et al.", 2020): DOI + "10.1038/s41586-020-2397-3",
    ("Widrow & Hoff", 1960): DOI + "10.21236/AD0241531",
    ("Nagumo & Noda", 1967): DOI + "10.1109/TAC.1967.1098599",
    ("Steil", 2004): DOI + "10.1109/IJCNN.2004.1380039",
    ("Sussillo & Abbott", 2009): DOI + "10.1016/j.neuron.2009.07.018",
    ("Yuenyong", 2014): "https://t2r2.star.titech.ac.jp/rrws/file/CTT100675129/ATD100000413/",
    ("Ohkubo & Inubushi", 2024): DOI + "10.1038/s41598-024-81880-3",
    ("Pathak et al.", 2017): DOI + "10.1063/1.5010300",
    ("Saunders, Gammerman & Vovk", 1998): "https://mlanthology.org/icml/1998/saunders1998icml-ridge/",
    ("Hermans & Schrauwen", 2012): DOI + "10.1162/NECO_a_00200",
    ("Dong et al.", 2020): "https://proceedings.neurips.cc/paper/2020/hash/c348616cd8a86ee661c7c98800678fad-Abstract.html",
    ("Gauthier et al.", 2021): DOI + "10.1038/s41467-021-25801-2",
    ("Hoerzer, Legenstein & Maass", 2014): DOI + "10.1093/cercor/bhs348",
    ("Legenstein et al.", 2010): DOI + "10.1523/JNEUROSCI.4284-09.2010",
    ("Fiete & Seung", 2006): DOI + "10.1103/PhysRevLett.97.048104",
    ("Chattopadhyay et al.", 2020): DOI + "10.5194/npg-27-373-2020",
    ("London & Häusser", 2005): DOI + "10.1146/annurev.neuro.28.061604.135703",
    ("Lukoševičius & Jaeger", 2009): DOI + "10.1016/j.cosrev.2009.03.005",
    ("Lukoševičius", 2012): DOI + "10.1007/978-3-642-35289-8_36",
    ("Tanaka et al.", 2019): DOI + "10.1016/j.neunet.2019.03.005",
}
VENUE_URLS = {
    "PLoS Comput. Biol.": "https://journals.plos.org/ploscompbiol/", "eLife": "https://elifesciences.org/",
    "Neural Networks": "https://www.sciencedirect.com/journal/neural-networks",
    "Bioacoustics": "https://www.tandfonline.com/journals/tbio20", "Current Biology": "https://www.cell.com/current-biology/home",
    "Proc. R. Soc. B": "https://royalsocietypublishing.org/journal/rspb",
    "Animal Behaviour": "https://www.sciencedirect.com/journal/animal-behaviour",
    "Neurocomputing": "https://www.sciencedirect.com/journal/neurocomputing", "Chaos": "https://pubs.aip.org/aip/cha",
    "Neural Comput.": "https://direct.mit.edu/neco", "Phys. Rev. E": "https://journals.aps.org/pre/",
    "Sci. Rep.": "https://www.nature.com/srep/",
    "Front. Comput. Neurosci.": "https://www.frontiersin.org/journals/computational-neuroscience",
    "Physics Reports": "https://www.sciencedirect.com/journal/physics-reports",
    "Science": "https://www.science.org/journal/science", "Nature": "https://www.nature.com/nature/",
    "Nature Commun.": "https://www.nature.com/ncomms/", "PNAS": "https://www.pnas.org/",
    "ICASSP": "https://2027.ieeeicassp.org/", "IJCNN": "https://ijcnn.org/2027/authors/call-for-papers", "ESANN": "https://www.esann.org/",
    "ICANN": "https://e-nns.org/icann2027/", "ICML": "https://icml.cc/Conferences/FutureMeetings",
    "NeurIPS": "https://neurips.cc/Conferences/FutureMeetings", "IJCAI": "https://www.ijcai.org/future_conferences",
    "COLT": "https://learningtheory.org/colt2027/index.html",
    "ICPR": "https://www.icpr2027.com/web/instructions", "EUSIPCO": "https://eusipco2027.de/calls/",
    "ECAI": "https://ecai2027.org/", "PRICAI": "https://2026.pricai.org/calls/call-for-papers",
    "ICONIP": "https://www.apnns.org/conferences/", "ICASSP 2028": "https://2028.ieeeicassp.org/",
}
NEXT = {"ICASSP": "ICASSP 2027, Toronto, 16-21 mai 2027 (soumissions closes)", "IJCNN": "IJCNN 2027, Le Cap, 14-18 juin 2027",
        "ICPR": "B (ICORE 2026). En ligne. Enregistrement le 1er mars 2027, article court le 8 mars (8 pages hors références).",
        "EUSIPCO": "B historique (CORE 2018). Darmstadt. Cinq pages, références comprises, pour le 6 février 2027.",
        "ECAI": "A (ICORE 2026). Athènes. Dépôt le 14 avril 2027. Viser une conclusion qui se vérifie sur plusieurs tâches.",
        "PRICAI": "B (ICORE 2026). Dépôt estimé vers le 13 juin 2027. En 2026, le format court comptait 6 à 11 pages.",
        "ICONIP": "B (ICORE 2026). Bangkok, novembre 2027. Dépôt estimé début mai.",
        "ICASSP 2028": "B historique (CORE 2023). Tokyo. Dépôt estimé vers le 23 septembre 2027.",
        "ESANN": "ESANN 2027 est close depuis le 25 novembre 2026 ; le site présente encore ESANN 2027, ESANN 2028 n'est pas annoncée",
        "COLT": "COLT 2027, Tokyo, 28 juin-2 juillet 2027 ; date limite le 29 janvier 2027",
        "IJCAI": "IJCAI 2027, Kyoto, 7-13 août 2027 ; date limite non publiée, en janvier les années passées",
        "ICANN": "ICANN 2027, Linköping, 24-27 août 2027 ; date limite pas encore publiée", "ICML": "ICML 2027, Rio de Janeiro, dates à venir",
        "NeurIPS": "NeurIPS 2027, Europe, ville et dates à venir"}


SHOWN = {"ESANN": "ESANN 2028"}  # (ESANN 2027 is closed)


def venue(name):
    level = PRESTIGE[name]
    marks = "".join(f'<i class="{"on" if k < level else ""}"></i>' for k in range(3))
    kind = "conférence" if name in CONFERENCES else "revue"
    rate = f' <i class="rate">{ACCEPTED[name]}</i>' if name in ACCEPTED else ""
    title = f"{kind}, {LEVELS[level]}" + (f" ; {NEXT[name]}" if name in NEXT else "")
    return (f'<a class="{"conf" if name in CONFERENCES else ""}" href="{VENUE_URLS[name]}" target="_blank" rel="noopener" title="{escape(title)}">'
            f'{escape(SHOWN.get(name, name))}{rate}<b class="lvl" aria-label="{LEVELS[level]}">{marks}</b></a>')


PAPERS = [
    dict(n=1, hue=5, family="canari", title="Un modèle de chant de canari réaliste", status="Générateur à améliorer", maturity=1,
         key=None,
         fig=flow([("écoute", False), ("réservoir", False), ("phrase suivante", True), ("mémoire", False), ("voix GAN", True)], loop="la syllabe produite est réécoutée"),
         idea="Un réservoir qui écoute le chant choisit la phrase suivante, une voix la chante, et il réécoute ce qu'elle chante. Le but : des chants à la syntaxe de canari.",
         angle="Un modèle de production du chant, évalué contre les vrais chants et contre une chaîne de Markov ; un stimulus pour des expériences de playback.",
         missing=["Battre la chaîne de Markov d'ordre 1 : distance des 3-grammes aux vrais chants 0,16, contre 0,11 pour elle.", "Brancher le GAN : la voix rejoue encore des phrases enregistrées.", "Une validation perceptive, si la revue l'attend."],
         refs=[("Markowitz et al.", 2013, "PLoS Comput. Biol.", "la syntaxe dépend de plusieurs phrases passées", False),
               ("Gardner, Naef & Nottebohm", 2005, "Science", "le chant imité jeune est réorganisé en phrasé de canari à l'âge adulte", False),
               ("Pagliarini et al.", 2021, "préprint HAL", "un WaveGAN de faible dimension pour les syllabes de canari", False),
               ("Trouvain & Hinaut", 2021, "ICANN", "un décodeur du chant de canari, ESN et LSTM", False),
               ("Cohen et al.", 2022, "eLife", "TweetyNet, annotation automatique des chants", False)],
         venues=["PLoS Comput. Biol.", "eLife", "Neural Networks", "ICASSP", "Bioacoustics"]),
    dict(n=2, hue=2, family="canari", title="Le canari « groove » : l'audio en dit plus que les étiquettes", status="Pari : Science ou Nature", maturity=2,
         key=("+7,2 pts", "de l'audio sur les étiquettes, en test"),
         fig=dots([("audio", 86.1, True, ""), ("syllabes", 81.0, False, ""), ("étiquettes", 79.0, False, ""), ("n-gramme", 78.0, False, "adaptatif")], 76, 88),
         idea="L'audio prédit la phrase suivante mieux que toute la suite des étiquettes : le chant porte une information syntaxique hors des étiquettes, peut-être dans son rythme.",
         angle="Une affirmation générale sur la syntaxe chantée : comme la prosodie en parole, le rythme du chant annonce la suite. Le modèle sert d'instrument de mesure.",
         missing=["Étiquettes avec durées (CANOPY_LABELS_WITH_DURATIONS), puis durées permutées.", "Plusieurs canaris : tout repose sur marron1.", "Exclure sous-types non annotés, erreurs d'annotation et position dans le chant.", "Un lien avec la prosodie humaine ou avec des enregistrements neuronaux."],
         refs=[("Markowitz et al.", 2013, "PLoS Comput. Biol.", "syntaxe à longue portée du canari", False),
               ("Cohen et al.", 2020, "Nature", "des états neuronaux cachés portent la syntaxe du canari", False),
               ("James, Sakata et al.", 2026, "Science", "allongement final et rythmes réguliers chez 360 espèces d'oiseaux et 52 langues", False),
               ("Roeske et al.", 2020, "Current Biology", "des rythmes catégoriels chez le rossignol progné, comme chez l'humain", False),
               ("Gardner et al.", 2005, "Science", "le phrasé du chant appris est réorganisé", False),
               ("Cohen et al.", 2022, "eLife", "annotations automatiques et leurs erreurs", False)],
         venues=["Science", "Nature", "Current Biology", "PNAS", "Nature Commun.", "eLife", "PLoS Comput. Biol.", "Proc. R. Soc. B", "Animal Behaviour"]),
    dict(n=3, hue=3, family="lecture", title="NLMS comme lecture de réservoir", status="Article court, ordre 1", maturity=2,
         key=("+5,0 pts", "de NLMS sur LMS sans fuite, sur marron1"),
         fig='<p class="figcap">avec fuite</p>' + dots([("RLS", 86.2, False, ""), ("ridge", 86.1, False, ""), ("NLMS", 80.8, True, ""), ("LMS", 80.5, False, "")], 74, 88)
         + '<p class="figcap">sans fuite</p>' + dots([("ridge", 84.2, False, ""), ("RLS", 81.8, False, ""), ("NLMS", 80.3, True, ""), ("LMS", 75.2, False, "")], 74, 88),
         idea="NLMS, oublié entre LMS et RLS en reservoir computing, est une lecture en ligne plus sûre que LMS : son pas ne dépend plus de la norme des états, pourvu qu'on les standardise.",
         angle="Un article court, limité à l'ordre 1 : NLMS contre LMS et RLS, le rôle de la standardisation. L'ordre 2 reste aux papiers 4 et 5.",
         missing=["LMS et RLS sur JapaneseVowels et InsectSound, où ils n'ont pas tourné.", "Lever le budget de 3 000 époques, atteint par toutes les études.", "Expliquer les 5 points qui restent sous ridge."],
         refs=[("Widrow & Hoff", 1960, "rapport, Stanford", "LMS", False),
               ("Nagumo & Noda", 1967, "IEEE Trans. Autom. Control", "l'origine de NLMS", False),
               ("Steil", 2004, "IJCNN", "BPDC, pas normalisé comme NLMS", False),
               ("Sussillo & Abbott", 2009, "Neuron", "FORCE, lecture en ligne avec RLS", False),
               ("Yuenyong", 2014, "thèse, Tokyo Tech", "NLMS perd face à RLS, sans être défini", False)],
         venues=["ICPR", "EUSIPCO", "ECAI", "ICASSP 2028", "ESANN", "IJCNN", "ICONIP", "PRICAI", "ICANN", "Neurocomputing"]),
    dict(n=4, hue=7, family="lecture", title="L'ordre 2 en forme duale, réponse au generalized readout", status="Mesures de temps à faire", maturity=3,
         key=("k = p₁ + (p₁² + p₂)/2", "le noyau exact de tous les produits xⱼxⱼ'"),
         fig=bars([("InsectSound", 2.5, True, "val. croisée"), ("marron1", 0.2, False, "test"), ("JapaneseVowels", 0.0, False, "test")], 3.6),
         idea="La lecture de tous les produits se calcule exactement par un noyau, sans former les N(N+1)/2 monômes ; ridge, NLMS et EH en ont une forme duale.",
         angle="L'exactitude et le coût : quand la forme duale gagne (m < 2D), sa mémoire, et quand l'ordre 2 aide vraiment.",
         missing=["Des temps mesurés, pas seulement des comptes d'opérations.", "Se situer face aux lectures à noyau publiées.", "Le gain dépend du jeu : nul sur JapaneseVowels."],
         refs=[("Ohkubo & Inubushi", 2024, "Sci. Rep.", "le generalized readout, en forme primale", False),
               ("Pathak et al.", 2017, "Chaos", "des états au carré dans la lecture, pour une sortie de Lorenz", False),
               ("Saunders, Gammerman & Vovk", 1998, "ICML", "ridge en variables duales", False),
               ("Hermans & Schrauwen", 2012, "Neural Comput.", "noyaux récurrents d'un ESN infini", False),
               ("Dong et al.", 2020, "NeurIPS", "reservoir computing et noyaux récurrents", False),
               ("Gauthier et al.", 2021, "Nat. Commun.", "NG-RC, monômes d'entrées retardées", False)],
         venues=["NeurIPS", "ICML", "IJCAI", "Chaos", "Neural Networks", "Neural Comput.", "Phys. Rev. E", "Sci. Rep.", "ESANN"]),
    dict(n=5, hue=1, family="lecture", title="L'ordre 2 avec des règles locales qui rattrapent RLS et ridge", status="Campagnes en cours", maturity=2,
         key=("−0,4 pt", "entre neh:2, guidée par une récompense, et ridge"),
         fig=dots([("RLS", 86.2, False, ""), ("ridge", 86.1, False, ""), ("nlms:2", 85.8, True, ""), ("neh:2", 85.7, True, "récompense")], 84, 87),
         idea="Un neurone de lecture qui apprend par une règle locale, avec l'erreur ou une seule récompense, atteint presque les moindres carrés s'il lit des produits d'états.",
         angle="La plausibilité biologique : plasticité locale et interactions multiplicatives. Le cœur local, ce sont les sélections en forme primale ; la forme duale n'est pas locale.",
         missing=["La fin des campagnes InsectSound.", "JapaneseVowels : neh:2 y perd 2 points sur neh d'ordre 1 ; les règles locales y égalent ridge dès l'ordre 1.", "Lever le budget de 3 000 époques à l'ordre 1."],
         refs=[("Hoerzer, Legenstein & Maass", 2014, "Cereb. Cortex", "EH, règle hebbienne modulée par récompense", False),
               ("Legenstein et al.", 2010, "J. Neurosci.", "une règle hebbienne pilotée par une récompense globale et le bruit", False),
               ("Fiete & Seung", 2006, "Phys. Rev. Lett.", "la perturbation de nœud", False),
               ("Sussillo & Abbott", 2009, "Neuron", "FORCE et RLS", False),
               ("Chattopadhyay et al.", 2020, "Nonlinear Process. Geophys.", "des produits d'unités voisines dans la lecture", False),
               ("London & Häusser", 2005, "Annu. Rev. Neurosci.", "le calcul dendritique", False)],
         venues=["PLoS Comput. Biol.", "Neural Comput.", "Front. Comput. Neurosci.", "NeurIPS", "IJCAI"]),
    dict(n=6, hue=6, family="revue", title="Une revue des lectures de réservoir", status="À rédiger en dernier", maturity=1,
         key=("3 jeux", "marron1, JapaneseVowels, InsectSound, une HPO par configuration"),
         fig=grid(["ridge", "RLS", "NLMS", "EH"], ["ordre 1", "partiel", "complet"],
                  {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)}),
         idea="Comparer, sur des réservoirs optimisés pour chacune, toutes les façons d'apprendre et de lire une sortie : règle, fenêtre, ordre et forme.",
         angle="Une revue avec un banc d'essai équitable, qui manque au domaine ; elle cite les papiers 3 à 5.",
         missing=["Les papiers 4 et 5, sur lesquels elle s'appuie."],
         refs=[("Lukoševičius & Jaeger", 2009, "Comput. Sci. Rev.", "la revue de référence du domaine", False),
               ("Lukoševičius", 2012, "", "le guide pratique des ESN", False),
               ("Tanaka et al.", 2019, "Neural Networks", "le reservoir computing physique", False),
               ("Gauthier et al.", 2021, "Nat. Commun.", "lectures non linéaires récentes", False)],
         venues=["Neural Networks", "Neurocomputing", "Chaos", "Physics Reports"]),
]

CSS = """
:root{--bg:#f6f5f1;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--rule:#e1e0d9;--track:#e9e8e2;
--h1:#2a78d6;--h2:#e2572b;--h3:#14916a;--h5:#b97700;--h6:#6f6d66;--h7:#6250d6;--warn:#9a6700}
@media (prefers-color-scheme: dark){:root{--bg:#121212;--card:#1d1d1c;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--rule:#383835;--track:#2c2c2a;
--h1:#3987e5;--h2:#d95926;--h3:#199e70;--h5:#b97700;--h6:#8a8880;--h7:#7c6fe0;--warn:#e3b341}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:1260px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:26px;margin:0 0 12px;letter-spacing:-.01em}.lead{color:var(--ink2);margin:0 0 18px;max-width:880px}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:0 0 20px;font-size:12.5px;color:var(--ink2);align-items:center}
.legend b{color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(370px,1fr));gap:18px}
.card{--c:var(--h1);background:var(--card);border:1px solid var(--rule);border-radius:14px;overflow:hidden;display:flex;flex-direction:column}
.band{background:var(--c);color:#fff;padding:14px 18px 12px;display:flex;gap:12px;align-items:flex-start}
.num{flex:none;font:700 30px/1 system-ui;opacity:.9}
.band h2{font-size:16.5px;line-height:1.3;margin:2px 0 6px}
.meta{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:12px}
.pill{background:rgba(255,255,255,.22);border-radius:999px;padding:2px 10px;font-weight:600}.fam{background:#fff;color:var(--c);border-radius:999px;padding:2px 10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;font-size:10.5px}.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:3px;vertical-align:-1px}
.meter{display:inline-flex;gap:3px}.meter i{width:14px;height:6px;border-radius:3px;background:rgba(255,255,255,.3)}.meter i.on{background:#fff}
.body{padding:14px 18px 16px;display:flex;flex-direction:column;gap:12px}
.key{display:flex;align-items:baseline;gap:10px;border-left:3px solid var(--c);padding-left:10px}
.key strong{font:600 24px/1.1 Georgia,serif;color:var(--c);white-space:nowrap}.key span{color:var(--ink2);font-size:13px}
.fig{width:100%;height:auto;display:block}.figcap{font-size:11.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:0 0 -2px}
.fig .grid{stroke:var(--rule);stroke-width:1}.fig .tick{fill:var(--muted);font-size:10.5px}
.fig .lab{font-size:11.5px}.fig .lab.hot,.fig .val.hot{fill:var(--ink);font-weight:600}.fig .lab.cold,.fig .val.cold{fill:var(--ink2)}
.fig .val{font-size:11.5px}.fig .stem{stroke-width:2;stroke-linecap:round}.fig .stem.hot{stroke:var(--c)}.fig .stem.cold{stroke:var(--track)}
.fig .dot.hot,.fig .bar.hot,.fig .cell.hot{fill:var(--c)}.fig .dot.cold,.fig .bar.cold{fill:var(--muted)}.fig .cell.cold{fill:var(--track)}
.fig .box{stroke-width:1.5}.fig .box.hot{fill:var(--c);stroke:var(--c)}.fig .box.cold{fill:none;stroke:var(--muted)}
.fig .boxt{font-size:10.5px}.fig .boxt.hot{fill:#fff;font-weight:600}.fig .boxt.cold{fill:var(--ink2)}
.fig .arrow{stroke:var(--muted);stroke-width:1.5}.fig .loop{stroke:var(--c);stroke-width:1.8}.fig .looplab{fill:var(--c);font-size:10.5px;font-weight:600}.fig .looplab-bg{fill:var(--card)}.fig .arrowhead{fill:var(--muted)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
h3{display:flex;align-items:center;gap:6px;font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:0 0 3px;font-weight:700}
h3::before{content:"";width:8px;height:8px;border-radius:2px;background:var(--c)}
p{margin:0}ul{margin:0;padding-left:16px}li{margin:2px 0}
.missing li::marker{content:"○  ";color:var(--c)}
.refs{list-style:none;padding:0;display:flex;flex-direction:column;gap:5px}
.refs li{display:grid;grid-template-columns:42px 1fr;gap:8px;align-items:baseline}
.year{font:600 11.5px/1.6 system-ui;color:var(--c);background:var(--track);border-radius:5px;text-align:center}
.refs b{font-weight:600}.refs i{color:var(--muted);font-style:normal}.refs .what{color:var(--ink2)}
.check{color:var(--warn);font-size:11px;font-weight:600}
.venues{display:flex;flex-wrap:wrap;gap:6px}.venues a{border:1px solid var(--c);color:var(--ink);border-radius:999px;padding:1px 10px;font-size:12px;display:inline-flex;align-items:center;gap:6px;text-decoration:none}.venues a.conf,.legend .conf{border-style:dashed}.refs a.ref{color:var(--ink);text-decoration:underline;text-decoration-color:var(--rule);text-underline-offset:2px}.legend .conf{border:1px dashed var(--ink2);border-radius:999px;padding:0 8px}.rate{font-style:normal;color:var(--muted);font-size:11px}.lvl{display:inline-flex;gap:2px}.lvl i{width:6px;height:6px;border-radius:50%;background:var(--track)}.lvl i.on{background:var(--c)}.legend .lvl i.on{background:var(--ink2)}
.sec{font-size:19px;margin:34px 0 2px}.secnote{color:var(--ink2);font-size:13px;margin:0 0 14px}.minis{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px}.mini{background:var(--card);border:1px solid var(--rule);border-left:4px solid var(--c);border-radius:10px;padding:12px 14px;display:flex;flex-direction:column;gap:8px;font-size:13px}.mtop{display:flex;gap:9px;align-items:baseline}.mnum{font:700 18px/1 system-ui;color:var(--c)}.mini h2{font-size:14px;line-height:1.3;margin:0}.mfam{align-self:flex-start;color:var(--c);border:1px solid var(--c);border-radius:999px;padding:0 8px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;font-size:10px}.mini h3{font-size:10.5px}.mini .venues a{font-size:11.5px}@media (max-width:480px){.two{grid-template-columns:1fr}.grid{grid-template-columns:1fr}}
"""


# further ideas, on smaller cards under the six papers: letter, hue, family, title, idea, what is in hand, what is
# missing, venues
IDEAS = [
    dict(n="A", hue=1, family="lecture", title="Pourquoi EH échoue hors du temps continu, et comment la réparer",
         idea="La règle EH de l'article suppose une sortie qui change lentement. Avec une mise à jour par exemple, il faut la perturbation de nœud et une récompense par sortie.",
         have="EH de l'article : 5,6 % et 6,2 %, au hasard (campagne exploratoire). Version corrigée : 81,6 % en test. Calcul : la règle suit le signe de l'erreur, plafonnée à σ/4.",
         missing="Une simulation qui confirme le calcul, et un second jeu de données.",
         venues=["Neural Comput.", "IJCNN"]),
    dict(n="B", hue=3, family="lecture", title="Comment lire un réservoir : points d'entraînement et mode de décision",
         idea="Entraîner et décider sur le dernier état, la fin de la phrase, sa moyenne ou toutes ses trames : un banc d'essai où chaque lecture a ses propres réservoirs.",
         have="24 lectures, avec et sans fuite, une HPO chacune : de 53,5 % à 86,1 % en test sur marron1.",
         missing="Un second jeu de données, et l'explication du classement.",
         venues=["Neural Networks", "ICANN"]),
    dict(n="C", hue=7, family="réservoir", title="La plasticité HAG remplace le réglage de la fuite",
         idea="Des réservoirs adaptés par plasticité homéostatique n'ont plus besoin d'un taux de fuite réglé pour la tâche.",
         have="Sans fuite, en test : 84,2 % pour HAG, contre 81,9 % et 79,4 % pour les réservoirs aléatoires. Avec fuite, tous entre 84,8 et 86,1 %.",
         missing="Vérifier ce que couvre déjà le papier de hag, et le montrer sur d'autres tâches.",
         venues=["Neural Comput.", "PLoS Comput. Biol."]),
    dict(n="M", hue=1, family="théorie", title="Chants ou neurones : le rang de Φ et l'arrêt précoce",
         idea="Faut-il plus d'exemples ou plus de neurones ? Ce que bornent le rang de Φ, la pénalité ridge et l'arrêt précoce d'une règle en ligne.",
         have="Rangs mesurés sur marron1 : 507, 2 294 et 6 764 pour 7 179 exemples. Les règles en ligne d'ordre 1 s'arrêtent toutes au budget de 3 000 époques.",
         missing="Un théorème, ou des courbes d'apprentissage selon le nombre de chants et de neurones.",
         venues=["COLT", "Neural Comput."]),
]


def nb(text):
    """The text with unbreakable spaces inside its numbers (6 764) and before % and :."""
    import re
    return re.sub(r"(?<=\d) (?=\d{3}|%)", "\u202f", escape(text))


def small(p):
    return f"""<section class="mini" style="--c:var(--h{p['hue']})">
<div class="mtop"><span class="mnum">{p['n']}</span><h2>{escape(p['title'])}</h2></div>
<span class="mfam">{escape(p['family'])}</span>
<p>{nb(p['idea'])}</p>
<div><h3>En main</h3><p>{nb(p['have'])}</p></div>
<div><h3>Ce qui manque</h3><p>{nb(p['missing'])}</p></div>
<div class="venues">{"".join(venue(v) for v in sorted(p['venues'], key=lambda v: -PRESTIGE[v]))}</div>
</section>"""


def card(p):
    meter = "".join(f'<i class="{"on" if k < p["maturity"] else ""}"></i>' for k in range(4))
    refs = "".join(
        f'<li><span class="year">{y}</span><span><a class="ref" href="{REF_URLS[(a, y)]}" target="_blank" rel="noopener"><b>{escape(a)}</b></a>{" <i>· " + escape(v) + "</i>" if v else ""}'
        f'<br><span class="what">{escape(w)}</span>{" <span class=check>à vérifier</span>" if chk else ""}</span></li>'
        for a, y, v, w, chk in p["refs"])
    return f"""<section class="card" style="--c:var(--h{p['hue']})">
<div class="band"><span class="num">{p['n']}</span><div><h2>{escape(p['title'])}</h2>
<div class="meta"><span class="fam">{escape(p['family'])}</span><span class="pill">{escape(p['status'])}</span><span class="meter" title="maturité {p['maturity']} sur 4">{meter}</span></div></div></div>
<div class="body">
{f'<div class="key"><strong>{escape(p["key"][0])}</strong><span>{escape(p["key"][1])}</span></div>' if p["key"] else ""}
{p['fig']}
<div><h3>Idée principale</h3><p>{escape(p['idea'])}</p></div>
<div class="two"><div><h3>Angle</h3><p>{escape(p['angle'])}</p></div>
<div><h3>Ce qui manque</h3><ul class="missing">{"".join(f"<li>{escape(m)}</li>" for m in p['missing'])}</ul></div></div>
<div><h3>Références</h3><ul class="refs">{refs}</ul></div>
<div><h3>Revues et conférences</h3><div class="venues">{"".join(venue(v) for v in sorted(p['venues'], key=lambda v: -PRESTIGE[v]))}</div></div>
</div></section>"""


page = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Six idées de papiers</title><style>{CSS}</style></head><body><main>
<h1>Six idées de papiers</h1>
<div class="legend"><span><b>Barre de maturité</b> : quatre crans, des résultats en main au papier prêt à rédiger.</span><span><b>Prestige</b> (estimation, pas un facteur d'impact) : <b class="lvl"><i class="on"></i><i class="on"></i><i class="on"></i></b> très sélectif · <b class="lvl"><i class="on"></i><i class="on"></i><i></i></b> référence du domaine · <b class="lvl"><i class="on"></i><i></i><i></i></b> spécialisé ou accessible. Le pourcentage d'une conférence est son taux d'acceptation récent.</span></div>
<div class="grid">
{"".join(card(p) for p in PAPERS)}
</div>
<h2 class="sec">Pistes</h2>
<p class="secnote">Des idées tirées des mêmes travaux, pas encore des projets.</p>
<div class="minis">
{"".join(small(p) for p in IDEAS)}
</div></main></body></html>
"""
open(OUT, "w").write(page)
print(OUT)

# (the home button of every page of visualisations/, added again since the page is rewritten)
import os, subprocess, sys
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "add_home.py"), os.path.dirname(OUT)], check=True)
