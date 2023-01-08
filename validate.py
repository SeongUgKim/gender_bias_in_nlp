import spacy
import pickle

def tokenize(src, lang):
    tokenizer = None
    ret = []
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
    # for i in range(src.shape[0]):
    for i in range(200000, 211230):
        temp = [token.text.lower() for token in word_tokenizer(src[lang][i])]
        ret.append((src[lang][i], temp, i))
    return ret


def select_non_empty_tokens(tokens_list, index_list):
    ret = []
    for tokens in tokens_list:
        if tokens[2] in index_list:
            if tokens[0] == '':
                continue
            ret.append((tokens[2], tokens[1]))
    return ret


def find_gender_sentences_index(tokens_list, male_words, female_words):
    index = []
    for tokens in tokens_list:
        for token in tokens[1]:
            if token in male_words or token in female_words:
                index.append(tokens[0])
                break
    return index


def validate(src_tokens_list, dest_tokens_list, src_male_words,
             src_female_words, dest_male_words, dest_female_words):
    index_list = [i for i in range(200000, 211230)]
    src_selected_tokens_list = select_non_empty_tokens(src_tokens_list, index_list)
    src_index_list = find_gender_sentences_index(src_selected_tokens_list, src_male_words, src_female_words)
    dest_selected_tokens_list = select_non_empty_tokens(dest_tokens_list, src_index_list)
    dest_index_list = find_gender_sentences_index(dest_selected_tokens_list, dest_male_words, dest_female_words)
    index = [dest_selected_tokens_list[i][0] for i in range(len(dest_selected_tokens_list))]
    non_gender = []
    gender = []
    for i in range(len(index)):
        if index[i] not in dest_index_list:
            non_gender.append(dest_tokens_list[index[i] - 200000][0])
        else:
            gender.append(dest_tokens_list[index[i] - 200000][0])
    with open('non_gender_de', 'wb') as f:
        pickle.dump(non_gender, f)
    with open('gender_de', 'wb') as f:
        pickle.dump(gender, f)
    return len(dest_index_list), len(dest_selected_tokens_list), len(dest_index_list) / len(dest_selected_tokens_list)
