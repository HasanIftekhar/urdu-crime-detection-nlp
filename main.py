from gensim.models import KeyedVectors
import re

# Load the pre-trained Urdu Word2Vec model
# Download from: https://github.com/jaleed96/urduvec
model = KeyedVectors.load_word2vec_format("urdu_wikipedia_vector300_word2vec_linguistic_variation_1.bin", binary=True)

# Test sentence in Urdu
preprocessed_text = "عراق اور شام نے اعلان کیا ہے دونوں ممالک جلد اپنے اپنے سفیروں کو واپس بغداد اور دمشق بھیج دیں گے"

# Extract Urdu words
input_sentence = re.findall(r'[\u0600-\u06ff]+', preprocessed_text)

# Print word vectors
for word in input_sentence:
    print(word, model.get_vector(word), "\n")

# Find similar words
print("Words similar to 'عراق':", model.most_similar("عراق"))
