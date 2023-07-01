# What is your favorite gender MLM?: Gender Bias Evaluation in Multilingual Masked Language Models

This GitHub page consists of the dataset and implementation of our paper "What is your favorite gender MLM?: Gender Bias Evaluation in Multilingual Masked Language Models."
Our work distinguishes itself from other works through its unique features and characteristics such as: 


Strengths
* It provides a multi-lingual gender lexicon in English, German, Spanish, Portuguese and Chinese. 
* It evaluates gender bias of language models on any corpus in these five languagues. 
* The evaluation corpus and the language model can be easily altered to assess gender bias.

## Guideline 

1. Multilingual Gender Lexicon

** MGL in five languages, English, German, Spanish, Portuguese, and Chinese are within eval_words folder in the repository.
** Encoded as a pickle file, each file is classified with respect to gender and language.
** In generating the pairs of sentences for evaluating gender bias of language models, each file is required as input.

2. Lexicon_based and Model_based Sentence Extraction

* Given this MGL from eval_words folder, lexicon_based and model_based sentence extraction is conducted through "extract.py" file.**  
* Within this file, one can change the evaluation corpus by modifying the arguments of this Python file.
* The required arguments to pass are the language of the corpus(model), male gender lexicon, female gender lexicon, and the corpus.
* This file first tokenizes the corpus, extracts the sentences containing the gendered word, generates the sentences, and writes the sentences in pickle format.
* One can also use Jupyter Notebook to make the sentence that is shown in "extraction_chn.ipynb" file.
* An illustration of how this pipeline works is shown in the main function of "extract.py" file.

3. Multilingual Bias Evaluation Metrics

* Using the sentences, Strict Bias Metrics that quantify gender bias of language models can be evaluated in "MBE_Calculation.ipynb" file.
* With the size of our corpus being approximately 30,000 sentences for each language, our evaluation for each language took less than 10 minutes for each language.

## Contact

* [Jeongrok Yu](https://www.emorynlp.org/bachelors/jeongrok-yu)
