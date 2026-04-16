from gensim.models import KeyedVectors
import re

model = KeyedVectors.load_word2vec_format("urdu_wikipedia_vector300_word2vec_linguistic_variation_1.bin",binary=True)
Preprocessed_text="عراق اور شام نے اعلان کیا ہے دونوں ممالک جلد اپنے اپنے سفیروں کو واپس بغداد اور دمشق بھیج دیں گے "

input_sentence = re.findall(r'[\u0600-\u06ff]+', Preprocessed_text)
print(input_sentence[0])
for i in range(len(input_sentence)):
    print(input_sentence[i],model.get_vector(input_sentence[i]),"\n")
model.




