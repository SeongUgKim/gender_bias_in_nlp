import argparse
import csv

import spacy
import pandas as pd
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True,
                        choices=['en', 'de', 'ja', 'ar', 'es', 'pt', 'ru', 'id', 'zh'],
                        help='Path to evaluation dataset.')
    parser.add_argument('--male_words', type=str, required=True)
    parser.add_argument('--female_words', type=str, required=True)
    parser.add_argument('--corpus', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)
    args = parser.parse_args()
    return args


def write_list(a_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(a_list, f)


def read_list(filename):
    with open(filename, 'rb') as f:
        n_list = pickle.load(f, encoding='utf8')
        return n_list


def tokenize(src, lang, k):
    tokenizer = None
    ret = []
    count = 0
    talks = []
    if lang == 'en':
        tokenizer = 'en_core_web_sm'
    elif lang == 'de':
        tokenizer = 'de_core_news_sm'
    elif lang == 'ja':
        tokenizer = 'ja_core_news_sm'
    elif lang == 'ar':
        tokenizer = 'xx_ent_wiki_sm'
    elif lang == 'es':
        tokenizer = 'es_core_news_sm'
    elif lang == 'pt':
        tokenizer = 'pt_core_news_sm'
    elif lang == 'ru':
        tokenizer = 'ru_core_news_sm'
    elif lang == 'id':
        tokenizer = 'xx_ent_wiki_sm'
    elif lang == 'zh':
        tokenizer = 'zh_core_web_sm'
    word_tokenizer = spacy.load(tokenizer)
    for i in range(10000, 12000):
        temp1 = [token.text.lower() for token in word_tokenizer(src[lang][i])]
        temp2 = [token.text for token in word_tokenizer(src[lang][i])]
        # if i > 0 and src['talkid'][i - 1] != src['talkid'][i] and count >= 2 * k + 1:
        #     ret.extend(talks)
        #     talks = []
        #     count = 0
        # else:
        #     talks.append((src['talkid'][i], src[lang][i], temp))
        ret.append((src['talkid'][i], src[lang][i], temp1, temp2, i))
    return ret


def get_index(tokens_list, index, k):
    minus = []
    plus = []
    current = []
    ret = []
    for i in reversed(range(1, k + 1)):
        if index - i < 0 or tokens_list[index - i][0] != tokens_list[index][0]:
            continue
        minus.append(index - i)
    # masking
    current.append(index)
    for i in range(1, k + 1):
        if index + i >= len(tokens_list) or tokens_list[index + i][0] != tokens_list[index][0]:
            continue
        plus.append(index + i)
    if len(minus) < k:
        for i in range(1, k - len(minus) + 1):
            plus.append(index + k + i)
    if len(plus) < k:
        for i in range(1, k - len(plus) + 1):
            minus.insert(0, index - (k + i))
    ret.extend(minus)
    ret.extend(current)
    ret.extend(plus)
    return ret


def extract_kth_neighbor_sentences(tokens_list, male_words, female_words, k):
    output = []
    for i in range(len(tokens_list)):
        male = [j for j in range(len(tokens_list[i][2])) if tokens_list[i][2][j] in male_words]
        female = [j for j in range(len(tokens_list[i][2])) if tokens_list[i][2][j] in female_words]
        # male = [word for word in tokens_list[i][2] if word in male_words]
        # female = [word for word in tokens_list[i][2] if word in female_words]
        if (len(male) == 1 and len(female) == 0) or (len(male) == 0 and len(female) == 1):
            sentence = ''
            indices = get_index(tokens_list, i, k)
            for index in indices:
                if i == index:
                    temp = tokens_list[index][1]
                    temp_word = tokens_list[index][3][male[0]] if len(male) == 1 else tokens_list[index][3][female[0]]
                    mask_index = temp.find(temp_word)
                    # if mask_index == -1:
                    #     temp_list = list(temp_word)
                    #     temp_list[0] = temp_list[0].upper()
                    #     temp_word = ''.join(temp_list)
                    #     mask_index = temp.find(temp_word)
                    # if mask_index == -1:
                    result = temp[:mask_index] + '[MASK]' + temp[mask_index + len(temp_word):]
                    sentence = sentence + (result + ' ')
                else:
                    sentence = sentence + (tokens_list[index][1] + ' ')
            output.append(sentence[:len(sentence) - 1])
    return output


def main(args):
    lang = args.lang
    male = args.male_words
    female = args.female_words
    corpus = args.corpus
    k = args.k
    ted = pd.read_csv(corpus, sep='\t', keep_default_na=False,
                      encoding='utf8', quoting=csv.QUOTE_NONE)
    if lang == 'zh':
        ted_lang = ted[['talkid', 'zh-cn']]
    else:
        ted_lang = ted[['talkid', lang]]
    male_words = read_list(male)
    female_words = read_list(female)
    tokens_list = tokenize(ted_lang, lang, k)
    gender_sentence = extract_kth_neighbor_sentences(tokens_list, male_words, female_words, k)
    write_list(gender_sentence, 'temp')
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
