import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#random example text
text = 'today is a good day. today is a sunny day, I want to do some sports and be healthier. Manbing is a beautiful and smart girl'
#sentence tokenization
sentences = nltk.sent_tokenize(text)
#word tokenization
words = [nltk.word_tokenize(s) for s in sentences]
print(sentences)
print(words)
#part-of-speech tagging
tagged_wt = [nltk.pos_tag(w) for w in words]
print(tagged_wt)
pattern_POS = []
for tag in tagged_wt:
    pattern_POS.append([v for k, v in tag])

print(pattern_POS)

#extracting nouns and verbs
nouns = []
for tag in tagged_wt:
    nouns.append([k for k, v in tag if v in ['NN', 'NNS', 'NNP', 'NNPS']])

verbs = []
for tag in tagged_wt:
    verbs.append([k for k, v in tag if 'V' in v])

from collections import Counter
import spacy
from tabulate import tabulate
nlp = spacy.load('en_core_web_lg')

doc = nlp(text)
noun_counter = Counter(token.lemma_ for token in doc if token.pos_ =='NOUN')
print(tabulate(noun_counter.most_common(5), headers=['Noun', 'Count']))

#dependency parsing
#(to understand the relationships between words in a sentence, a more fine grained attribute)
doc = nlp(sentences[2])
spacy.displacy.render(doc, style='dep', options={'distance' : 140}, jupyter=False)

#NER (can be recongize names type that are not English names)
doc = nlp(u"My name is Robin and I live now in Aachen.")
entity_types = ((ent.text, ent.label_) for ent in doc.ents)
print(tabulate(entity_types, headers=['Entity', 'Entity_types']))

#Building conversational bots


