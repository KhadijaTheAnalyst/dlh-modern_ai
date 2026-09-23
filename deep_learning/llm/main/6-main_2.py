#!/usr/bin/env python3

translate_text = __import__('6-translate_text').translate_text

translator_fr_en = translate_text("Helsinki-NLP/opus-mt-fr-en")
sentence1 = "L'intelligence artificielle transforme le monde."
sentence2 = "Les modèles d'apprentissage automatique s'améliorent avec le temps."

translations = translator_fr_en([sentence1, sentence2])
for i, t in enumerate(translations, 1):
    print(f"Sentence {i} translation:", t['translation_text'])
