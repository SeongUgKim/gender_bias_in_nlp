import argparse
import csv

import spacy
import pandas as pd
import pickle
from transformers import pipeline
from transformers import BertTokenizer
import random

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
    for i in range(src.shape[0]):
        # do not need if the language does not contain upper case and lower case ex) Chinese
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


def rule_based_extract(tokens_list, male_words, female_words, k):
    male_output = []
    female_output = []
    cnt = 0
    for i in range(len(tokens_list)):
        male = [j for j in range(len(tokens_list[i][2])) if tokens_list[i][2][j] in male_words]
        female = [j for j in range(len(tokens_list[i][2])) if tokens_list[i][2][j] in female_words]
        if (len(male) == 1 and len(female) == 0) or (len(male) == 0 and len(female) == 1):
            male_sentence = ''
            female_sentence = ''
            indices = get_index(tokens_list, i, k)
            for index in indices:
                if i == index:
                    temp = tokens_list[index][1]
                    if len(male) == 1:
                        temp_word = tokens_list[index][3][male[0]]
                        idx_within_lexicon = male_words.index(tokens_list[index][2][male[0]])
                        corresponding_word = female_words[idx_within_lexicon]
                        genderword_idx = temp.find(temp_word)
                        if temp_word[0].isupper():
                            corresponding_word = corresponding_word.replace(corresponding_word[0], corresponding_word[0].upper())
                        corresponding_sentence = temp[:genderword_idx] + corresponding_word + temp[genderword_idx + len(temp_word):]
                        male_sentence = male_sentence + temp + ' '
                        female_sentence = female_sentence + corresponding_sentence + ' '
                    else:
                        temp_word = tokens_list[index][3][female[0]]
                        idx_within_lexicon = female_words.index(tokens_list[index][2][female[0]])
                        corresponding_word = male_words[idx_within_lexicon]
                        genderword_idx = temp.find(temp_word)
                        if temp_word[0].isupper():
                            corresponding_word = corresponding_word.replace(corresponding_word[0], corresponding_word[0].upper())
                        corresponding_sentence = temp[:genderword_idx] + corresponding_word + temp[genderword_idx + len(temp_word):]
                        male_sentence = male_sentence + temp + ' '
                        female_sentence = female_sentence + corresponding_sentence + ' '
                else:
                    male_sentence = male_sentence + (tokens_list[index][1] + ' ')
                    female_sentence = female_sentence + (tokens_list[index][1] + ' ')
            male_output.append(male_sentence[:len(male_sentence) - 1])
            female_output.append(female_sentence[:len(female_sentence) - 1])
            cnt += 1

    return male_output, female_output, cnt

def no_sie_rule_based_extract(tokens_list, male_words, female_words, k):
    male_output = {}
    female_output = {}
    male_idx = []
    female_idx = []
    cnt, n = 0, 0
    for i in range(len(tokens_list)):
        if 'er' in tokens_list[i][2]:
            continue
        male = [j for j in range(len(tokens_list[i][2])) if tokens_list[i][2][j] in male_words]
        female = [j for j in range(len(tokens_list[i][2])) if tokens_list[i][2][j] in female_words]
        if (len(male) == 1 and len(female) == 0) or (len(male) == 0 and len(female) == 1):
            male_sentence = ''
            female_sentence = ''
            indices = get_index(tokens_list, i, k)
            for index in indices:
                if i == index:
                    temp = tokens_list[index][1]
                    if len(male) == 1:
                        male_idx.append(i)
                        temp_word = tokens_list[index][3][male[0]]
                        idx_within_lexicon = male_words.index(tokens_list[index][2][male[0]])
                        corresponding_word = female_words[idx_within_lexicon]
                        genderword_idx = temp.find(temp_word)
                        if temp_word[0].isupper():
                            corresponding_word = corresponding_word.replace(corresponding_word[0], corresponding_word[0].upper())
                        corresponding_sentence = temp[:genderword_idx] + corresponding_word + temp[genderword_idx + len(temp_word):]
                        male_sentence = male_sentence + temp + ' '
                        female_sentence = female_sentence + corresponding_sentence + ' '
                    else:
                        female_idx.append(i)
                        temp_word = tokens_list[index][3][female[0]]
                        idx_within_lexicon = female_words.index(tokens_list[index][2][female[0]])
                        corresponding_word = male_words[idx_within_lexicon]
                        genderword_idx = temp.find(temp_word)
                        if temp_word[0].isupper():
                            corresponding_word = corresponding_word.replace(corresponding_word[0], corresponding_word[0].upper())
                        corresponding_sentence = temp[:genderword_idx] + corresponding_word + temp[genderword_idx + len(temp_word):]
                        male_sentence = male_sentence + corresponding_sentence + ' '
                        female_sentence = female_sentence + temp + ' '
                else:
                    male_sentence = male_sentence + (tokens_list[index][1] + ' ')
                    female_sentence = female_sentence + (tokens_list[index][1] + ' ')
            male_output[i] = male_sentence[:len(male_sentence) - 1]
            female_output[i] = female_sentence[:len(female_sentence) - 1]
            cnt += 1

    #after male and female sentences added
    if len(male_idx) > len(female_idx):
        n = len(male_idx) - len(female_idx)
        # toBeRemoved = male_idx[-n:]
        # del male_idx[-n:]
        toBeRemoved = random.sample(male_idx, n)

        #sanity check
        for i in toBeRemoved:
            male_idx.remove(i)

        for i in toBeRemoved:
            del male_output[i]
            del female_output[i]
    else:
        n = len(female_idx) - len(male_idx)
        toBeRemoved = random.sample(female_idx, n)

        #sanity check
        for i in toBeRemoved:
            female_idx.remove(i)

        for i in toBeRemoved:
            del male_output[i]
            del female_output[i]

    male_output = list(male_output.values())
    female_output = list(female_output.values())

    return male_output, female_output, cnt

def model_based_extract(masked_sentence_list, male_words, female_words, model):
    unmasker = pipeline('fill-mask', model=model, top_k=10)
    tokenizer = BertTokenizer.from_pretrained(model)
    result = []
    for sentence in masked_sentence_list:
        if len(tokenizer.tokenize(sentence)) < 512:
            result.append(unmasker(sentence))
    gender_predict = []
    for predicts in result:
        m = []
        f = []
        for predict in predicts:
            if predict['token_str'].lower() in male_words and len(m) == 0:
                m.append((predict['token_str'], predict['sequence']))
            if predict['token_str'].lower() in female_words and len(f) == 0:
                f.append((predict['token_str'], predict['sequence']))
        temp = {'male': m[0] if len(m) == 1 else None, 'female': f[0] if len(f) == 1 else None}
        gender_predict.append(temp)
    male = []
    female = []
    both = 0
    one = 0
    non = 0
    cnt = 0
    for predict in gender_predict:
        cnt += 1
        if predict['male'] is None and predict['female'] is None:
            non += 1
            continue
        if predict['male'] is not None and predict['female'] is not None:
            male.append((predict['male'][1]))
            female.append(predict['female'][1])
            both += 1
            continue
        one += 1
        sentence = predict['male'][1] if predict['female'] is None else predict['female'][1]
        word = predict['male'][0] if predict['female'] is None else predict['female'][0]
        counter_word_index = male_words.index(word.lower()) if predict['female'] is None else female_words.index(word.lower())
        counter_char_list = list(female_words[counter_word_index]) if predict['female'] is None else list(male_words[counter_word_index])
        counter_word = ''.join(counter_char_list)
        if word != counter_word:
            for i in range(len(list(word))):
                if 'A' <= list(word)[i] <= 'Z':
                    counter_char_list[i] = counter_char_list[i].upper()
            counter_word = ''.join(counter_char_list)
        index = sentence.find(word)
        counterpart = sentence[:index] + counter_word + sentence[index + len(word):]
        if predict['male'] is None:
            female.append(sentence)
            male.append(counterpart)
        else:
            male.append(sentence)
            female.append(counterpart)
    loss_prob = f'both: {both / cnt}, one: {one / cnt}, non: {non / cnt}, totalModel: {cnt}'
    return male, female, loss_prob


def extract_kth_neighbor_sentences(tokens_list, male_words, female_words, k):
    output = []
    for i in range(len(tokens_list)):
        male = [j for j in range(len(tokens_list[i][2])) if tokens_list[i][2][j] in male_words]
        female = [j for j in range(len(tokens_list[i][2])) if tokens_list[i][2][j] in female_words]
        if (len(male) == 1 and len(female) == 0) or (len(male) == 0 and len(female) == 1):
            sentence = ''
            indices = get_index(tokens_list, i, k)
            for index in indices:
                if i == index:
                    temp = tokens_list[index][1]
                    temp_word = tokens_list[index][3][male[0]] if len(male) == 1 else tokens_list[index][3][female[0]]
                    mask_index = temp.find(temp_word)
                    result = temp[:mask_index] + '[MASK]' + temp[mask_index + len(temp_word):]
                    sentence = sentence + (result + ' ')
                else:
                    sentence = sentence + (tokens_list[index][1] + ' ')
            output.append(sentence[:len(sentence) - 1])
    return output

def no_sie_extract_kth_neighbor_sentences(tokens_list, male_words, female_words, k):
    output = []
    for i in range(len(tokens_list)):
        if 'er' in tokens_list[i][2]:
            continue
        male = [j for j in range(len(tokens_list[i][2])) if tokens_list[i][2][j] in male_words]
        female = [j for j in range(len(tokens_list[i][2])) if tokens_list[i][2][j] in female_words]
        if (len(male) == 1 and len(female) == 0) or (len(male) == 0 and len(female) == 1):
            sentence = ''
            indices = get_index(tokens_list, i, k)
            for index in indices:
                if i == index:
                    temp = tokens_list[index][1]
                    temp_word = tokens_list[index][3][male[0]] if len(male) == 1 else tokens_list[index][3][female[0]]
                    mask_index = temp.find(temp_word)
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
    model = ''
    if lang == 'de':
        model = 'deepset/gbert-base'
    if lang == 'es':
        model = 'dccuchile/bert-base-spanish-wwm-uncased'
    if lang == 'pt':
        model = 'neuralmind/bert-base-portuguese-cased'
    if lang == 'en':
        model = 'bert-base-cased'
    if lang == 'zh':
        model = 'hfl/chinese-bert-wwm-ext'
    ted = pd.read_csv(corpus, sep='\t', keep_default_na=False,
                      encoding='utf8', quoting=csv.QUOTE_NONE)
    if lang == 'zh':
        ted_lang = ted[['talkid', 'zh-cn']]
    else:
        ted_lang = ted[['talkid', lang]]
    male_words = read_list(male)
    female_words = read_list(female)
    tokens_list = tokenize(ted_lang, lang, k)
    # rule_based_male, rule_based_female, cnt = rule_based_extract(tokens_list, male_words, female_words, k)
    rule_based_male, rule_based_female, cnt = no_sie_rule_based_extract(tokens_list, male_words, female_words, k)
    male_filename = 'sentence/modified_rule_based_male_sentences_' + lang
    female_filename = 'sentence/modified_rule_based_female_sentences_' + lang
    write_list(rule_based_male, male_filename)
    write_list(rule_based_female, female_filename)

    gender_sentence = extract_kth_neighbor_sentences(tokens_list, male_words, female_words, k)
    # gender_sentence = no_sie_extract_kth_neighbor_sentences(tokens_list, male_words, female_words, k)
    model_based_male, model_based_female, loss_prob = model_based_extract(gender_sentence, male_words, female_words, model)
    male_filename1 = 'sentence/model_based_male_sentences_' + lang
    female_filename1 = 'sentence/model_based_female_sentences_' + lang
    prob_filename = 'sentence/prob_' + lang + '.txt'
    write_list(model_based_male, male_filename1)
    write_list(model_based_female, female_filename1)
    loss_prob += f' totalRule: {cnt}'
    with open(prob_filename, 'w') as f:
        f.write(loss_prob)
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
