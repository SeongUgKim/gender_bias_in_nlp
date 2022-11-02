import pandas as pd
import pickle


def write_list(a_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(a_list, f)


def read_list(filename):
    with open(filename, 'rb') as f:
        n_list = pickle.load(f, encoding='utf8')
        return n_list


def remove_neutral_words(gender_words, neutral_words):
    for item in neutral_words:
        if item in gender_words:
            gender_words.remove(item)
    return gender_words


def main():
    with open('male_word_file.txt') as f:
        lines = f.readlines()
    male_words_en = [line[:-1] for line in lines]
    with open('female_word_file.txt') as f:
        lines = f.readlines()
    female_words_en = [line[:-1] for line in lines]
    lexicon_de = pd.read_csv('Languages CSV - German (1) -Final.csv')
    lexicon_es = pd.read_csv('SpanishFinal - Final.csv')
    lexicon_pt = pd.read_csv('Portuguese - Final.csv')
    male_words_de = [word.lower() for word in lexicon_de['Male Words'].tolist()]
    female_words_de = [word.lower() for word in lexicon_de['Female Words'].tolist()]
    neutral_words_de = [word.lower() for word in lexicon_de['Neutral Words'].tolist() if type(word) is str]
    male_words_es = lexicon_es['M'].tolist()
    female_words_es = lexicon_es['F'].tolist()
    neutral_words_es = [word for word in lexicon_es['Neutral'].tolist() if type(word) is str]
    male_words_pt = lexicon_pt['M'].tolist()
    female_words_pt = lexicon_pt['F'].tolist()
    neutral_words_pt = [word for word in lexicon_pt['Neutral'].tolist() if type(word) is str]
    only_male_de = remove_neutral_words(male_words_de, neutral_words_de)
    only_female_de = remove_neutral_words(female_words_de, neutral_words_de)
    only_male_es = remove_neutral_words(male_words_es, neutral_words_es)
    only_female_es = remove_neutral_words(female_words_es, neutral_words_es)
    only_male_pt = remove_neutral_words(male_words_pt, neutral_words_pt)
    only_female_pt = remove_neutral_words(female_words_pt, neutral_words_pt)
    write_list(male_words_en, 'eval_words/only_male_words_en')
    write_list(female_words_en, 'eval_words/only_female_words_en')
    write_list(only_male_de, 'eval_words/only_male_words_de')
    write_list(only_female_de, 'eval_words/only_female_words_de')
    write_list(only_male_es, 'eval_words/only_male_words_es')
    write_list(only_female_es, 'eval_words/only_female_words_es')
    write_list(only_male_pt, 'eval_words/only_male_words_pt')
    write_list(only_female_pt, 'eval_words/only_female_words_pt')


if __name__ == "__main__":
    main()
