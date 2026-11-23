"""
Generation of canary songs, phrase by phrase: the predictor of the next phrase (canopy.studies.next_phrase.task) chooses
the next phrase of the song that its reservoir hears, a working memory holds this choice while the phrase is sung, and
a voice sings it, heard in turn by the reservoir (canopy.studies.song_generation.generator). The voices, in stages: recorded
phrases of the train songs (stage 1, the loop without the GAN), then the syllables of the GAN
(canopy.studies.syllable_generation, the latent command of the type in memory from the inverse model of
canopy.studies.syllable_generation.sensorimotor), then the gate of the memory opened by the readout of the phrase boundaries.
"""
