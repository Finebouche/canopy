"""
Generation of canary syllables: the low-dimensional WaveGAN of Pagliarini, Trouvain, Leblois & Hinaut (2021), "What
does the canary say? Low-dimensional GAN applied to birdsong" (original code:
https://github.com/spagliarini/low-dimensional-canary-GAN, TensorFlow 1), reimplemented
(canopy.syllable_generation.wavegan, after the PyTorch version of N. Trouvain, https://github.com/nTrouvain/canarygan),
trained on single syllables cut from the phrases of marron1 (canopy.syllable_generation.syllables), the song of the
next-phrase prediction: a generator of syllables from a 3-dimensional latent space, to be driven later by the
predictor of the next phrase (canopy.next_phrase_prediction).
"""
