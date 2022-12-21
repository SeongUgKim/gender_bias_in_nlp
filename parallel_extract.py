import pandas as pd
import csv
import validate
import pickle

def read_list(filename):
    with open(filename, 'rb') as f:
        n_list = pickle.load(f, encoding='utf8')
        return n_list
    
def write_list(a_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(a_list, f)
    
def create_word_list():
    words = pd.read_csv('Bolukbasi Gendered Words + CrowS Names - names.csv')
    male = words['Male'].values.tolist()
    female = words['Female'].values.tolist()
    name = words['Names'].values.tolist()
    male += female
    male += name
    res = [item.lower() for item in male if type(item) == str]
    return res

def extract_index(src, lang, word_list):    
    src_tokens_list = validate.tokenize(src, lang)
    index_list = []
    for tokens in src_tokens_list:
        cnt = 0
        for token in tokens[1]:
            if token in word_list:
                cnt += 1
        if cnt == 1:
            index_list.append(tokens[2])
    return index_list

def extract_sentence(df, lang, index_list):
    ret = [df[lang][i] for i in index_list if df[lang][i] != '']
    return ret
        
            

def main():
    ted_df = pd.read_csv('ted2020.tsv.gz', sep='\t', keep_default_na=False, encoding='utf8', quoting=csv.QUOTE_NONE)
    en = ted_df[['talkid', 'en']]
    word_list = create_word_list()
    index_list = extract_index(en, 'en', word_list)
    lang_list = ['en', 'de', 'es', 'pt']
    for lang in lang_list:
        sentence_list = extract_sentence(ted_df, lang, index_list)
        file_name = 'kaneko/sentences_' + lang
        write_list(sentence_list, filename=file_name)


if __name__ == "__main__":
    main()