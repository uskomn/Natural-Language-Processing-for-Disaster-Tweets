import random
import nltk
import pandas as pd
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").lower()
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(sentence, n=2):
    words = sentence.split()
    new_words = words.copy()
    random.shuffle(words)
    num_replaced = 0

    for word in words:
        synonyms = get_synonyms(word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if w == word else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return " ".join(new_words)

def random_deletion(sentence, p=0.1):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    return " ".join([w for w in words if random.uniform(0,1) > p])

def random_swap(sentence, n=2):
    words = sentence.split()
    if len(words) < 2:
        return sentence  # 直接返回，不做增强
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)



def random_augment(sentence):
    ops = [synonym_replacement, random_swap, random_deletion]
    op = random.choice(ops)
    try:
        if op == synonym_replacement:
            return op(sentence, n=2)
        elif op == random_swap:
            return op(sentence, n=2)
        elif op == random_deletion:
            return op(sentence, p=0.1)
    except Exception as e:
        print(f"增强失败：{e}，原句返回")
        return sentence



def augment_texts(texts, targets, times=1):
    """对 texts 增强若干倍"""
    aug_texts = []
    aug_targets = []

    for i in range(len(texts)):
        for _ in range(times):
            aug_text = random_augment(texts.iloc[i])
            aug_texts.append(aug_text)
            aug_targets.append(targets.iloc[i])

    # 返回 DataFrame 格式
    return pd.DataFrame({
        'text': aug_texts,
        'target': aug_targets
    })
