from urduhack import normalize
from urduhack.tokenization import word_tokenizer
from urduhack.preprocessing import normalize_whitespace
from urduhack.preprocessing import remove_punctuation
from urduhack.preprocessing import remove_accents
from urduhack.preprocessing import replace_urls
from urduhack.preprocessing import replace_emails
from urduhack.preprocessing import replace_phone_numbers
from urduhack.preprocessing import replace_numbers
from urduhack.preprocessing import replace_currency_symbols
from urduhack.preprocessing import remove_english_alphabets
from typing import FrozenSet
import re
import json
from urduhack.config import LEMMA_LOOKUP_TABLE_PATH
from sklearn.feature_extraction.text import CountVectorizer



from gensim.models import KeyedVectors


'''STEP 1
Sentence Tokenization
'''
sent = 'عراق اور شام نے اعلان کیا ہے دونوں ممالک جلد اپنے اپنے سفیروں کو واپس بغداد اور دمشق بھیج دیں گے ؟'
Tokenized_sentence=word_tokenizer(sent)
'''STEP 2
Sentence Preprocessing
1.Normalzie WhiteSpace
2.Remove Punctuation
3.Remove accents
4.Replace urls
5.Replace emails
6.Replace numbers
7.Replace phone numbers
8.Replace currancy symbols
9.Remove English alphabhets
#tokenized sentence into string for preprocessing 
'''
String_for_preprocessing = ' '.join([str(elem) for elem in Tokenized_sentence])
normalized_text = normalize_whitespace(String_for_preprocessing)
punctuation_Removed=remove_punctuation(normalized_text)
Removed_accents=remove_accents(punctuation_Removed)
Replaced_urls=replace_urls(Removed_accents)
Replaced_emails=replace_emails(Replaced_urls)
Replaced_phonenumbers=replace_phone_numbers(Replaced_emails)
Replaced_numbers=replace_numbers(Replaced_phonenumbers)
Replaced_currency=replace_currency_symbols(Replaced_numbers)
Removed_english_alpha=remove_english_alphabets(Replaced_currency)
Preprocessed_text=Removed_english_alpha


'''STEP 3
Stop words removal
'''
my_file = open("stopwords-ur.txt", "r",encoding='utf-8')
content = my_file.read()
Stopwords=content.split()

input_sentence = re.findall(r'[\u0600-\u06ff]+', Preprocessed_text)
Stopwords_removed = [word for word in input_sentence if not word in Stopwords]




'''STEP 4
Lemmatization
'''
_WORD2LEMMA = None


def lemma_lookup(text: str, lookup_path: str = LEMMA_LOOKUP_TABLE_PATH) -> list:


    tokens = text
    global _WORD2LEMMA
    if _WORD2LEMMA is None:
        with open(lookup_path, "r", encoding="utf-8") as file:
            _WORD2LEMMA = json.load(file)

    return [ _WORD2LEMMA[word] if word in _WORD2LEMMA else word for word in tokens]


print("Original Text :\n",sent)
print("Text after Preprocessing:\n",lemma_lookup(Stopwords_removed))

'''STEP 5
Vertorization
'''











