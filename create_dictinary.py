import re
from unidecode import unidecode

# Translation between diacritic and non-diacritic letters
DIACR_TO_NODIACR = str.maketrans("áčďéěíňóřšťúůýž", "acdeeinorstuuyz")

# Define dictionary for diacritic letters
words_dict = {}

with open('vesmir_articles37.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    text = text.lower()

    # Split text into words
    words = text.split()

    for word in words:
        word_without_diacritics = word.translate(DIACR_TO_NODIACR)

        if word_without_diacritics in words_dict:
            words_dict[word_without_diacritics].add(word)
        else:
            words_dict[word_without_diacritics] = {word}

# Save dictionary to file
with open('dict.txt', 'w', encoding='utf-8') as f:
    for key, value in words_dict.items():
        f.write(f'{key}: {", ".join(value)}\n')
