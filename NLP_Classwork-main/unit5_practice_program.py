
# ============================================================
# INSTALLATION (run once in Colab / terminal)
# pip install nltk transformers torch sentencepiece googletrans==4.0.0-rc1 sacremoses
# ============================================================

# ============================================================
# Program 1: Simple Rule-Based Chatbot using NLTK
# ============================================================

import nltk
from nltk.chat.util import Chat, reflections

nltk.download('punkt', quiet=True)

# Define conversation patterns
pairs = [
    (r"hi|hello|hey",
     ["Hello! How can I help you?",
      "Hi there! What can I do for you?"]),

    (r"what is your name?",
     ["I am Jason, your virtual assistant.",
      "My name is Jason!"]),

    (r"how are you?",
     ["I'm doing great, thank you!",
      "I'm fine. How about you?"]),

    (r"what can you do?",
     ["I can answer basic questions and have a conversation!"]),

    (r"tell me about (.*)",
     ["Sure! %1 is a very interesting topic.",
      "I know a bit about %1! What would you like to know?"]),

    (r"quit|bye|exit",
     ["Goodbye! Have a great day!",
      "Bye! Come back soon."]),

    (r"(.*)",
     ["I'm not sure I understand. Could you rephrase?",
      "Interesting! Tell me more."]),
]

def chatbot():
    print("Chatbot: Hello! I am Jason. Type 'quit' to exit.")
    chat = Chat(pairs, reflections)
    chat.converse()

if __name__ == "__main__":
    chatbot()

# ============================================================
# Program 2: AI Chatbot using DialoGPT (Transformers)
# ============================================================

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model     = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

print("DialoGPT Chatbot started. Type 'quit' to exit.\n")

chat_history_ids = None   # stores conversation context

for step in range(6):     # allow 6 conversation turns
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye! Have a nice day!")
        break

    # Encode new user input + EOS token
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token, return_tensors="pt"
    )

    # Append to chat history
    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None else new_input_ids
    )

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids, max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3, do_sample=True,
        top_k=50, top_p=0.92, temperature=0.75,
    )

    # Decode only the new response
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )
    print(f"Bot: {response}")

# ============================================================
# Program 3: Extractive Question Answering using BERT
# ============================================================

from transformers import pipeline

# Load pre-trained BERT QA pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="bert-large-uncased-whole-word-masking-finetuned-squad"
)

# -- Example 1: NLP Context -----------------------------------------------
context1 = """
Natural Language Processing (NLP) is a subfield of artificial intelligence
that focuses on the interaction between computers and human language.
NLP techniques are used in applications such as speech recognition, machine
translation, sentiment analysis, chatbots, and text summarization.
BERT (Bidirectional Encoder Representations from Transformers) is a
transformer-based model developed by Google in 2018. It is pre-trained on
large text corpora and can be fine-tuned for various NLP tasks including
question answering.
"""

questions1 = [
    "What is NLP?",
    "Who developed BERT?",
    "When was BERT developed?",
    "What tasks can BERT be fine-tuned for?",
]

print("=" * 60)
print(" EXAMPLE 1 -- Natural Language Processing")
print("=" * 60)
for q in questions1:
    result = qa_pipeline(question=q, context=context1)
    print(f"Q: {q}")
    print(f"A: {result['answer']}")
    print(f"   Confidence: {result['score']:.4f}")
    print("-" * 40)

# -- Example 2: Machine Learning Context ----------------------------------
context2 = """
Machine Learning is a branch of artificial intelligence that enables computers
to learn from data and improve their performance without being explicitly
programmed. Supervised learning uses labeled data to train models, while
unsupervised learning discovers hidden patterns in unlabeled data.
Deep Learning is a subset of machine learning that uses neural networks with
many layers to learn complex patterns from large amounts of data.
"""

questions2 = [
    "What is machine learning?",
    "What does supervised learning use?",
    "What is deep learning?",
]

print("\n" + "=" * 60)
print(" EXAMPLE 2 -- Machine Learning")
print("=" * 60)
for q in questions2:
    result = qa_pipeline(question=q, context=context2)
    print(f"Q: {q}")
    print(f"A: {result['answer']}")
    print(f"   Confidence: {result['score']:.4f}")
    print("-" * 40)

# ============================================================
# Program 4: Extractive Text Summarization using NLTK
# ============================================================

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq
import re

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

def extractive_summarize(text, num_sentences=3):
    # Step 1: Clean text
    clean_text = re.sub(r'\s+', ' ', text)
    clean_text = re.sub(r'[^a-zA-Z0-9. ]', '', clean_text)

    # Step 2: Tokenize
    sentences = sent_tokenize(text)
    words     = word_tokenize(clean_text.lower())
    stop_words = set(stopwords.words('english'))
    filtered   = [w for w in words if w not in stop_words and w.isalnum()]

    # Step 3: Word frequency (normalized)
    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    max_freq = max(freq.values()) if freq else 1
    freq = {w: f / max_freq for w, f in freq.items()}

    # Step 4: Score sentences
    scores = {}
    for sent in sentences:
        for w in word_tokenize(sent.lower()):
            if w in freq:
                scores[sent] = scores.get(sent, 0) + freq[w]

    # Step 5: Select top-N (preserve order)
    best    = heapq.nlargest(num_sentences, scores, key=scores.get)
    summary = ' '.join([s for s in sentences if s in best])
    return summary


text = """
Artificial Intelligence (AI) is transforming the modern world in unprecedented ways.
Machine Learning, a subset of AI, allows computers to learn from large amounts of
data without being explicitly programmed. Deep Learning, using multi-layered neural
networks, has achieved remarkable results in image recognition, natural language
processing, and speech recognition. Natural Language Processing (NLP) is a critical
branch of AI that enables machines to understand, interpret, and generate human
language. Applications of NLP include machine translation, sentiment analysis,
chatbots, and text summarization. Text summarization helps users quickly grasp the
key points of a document without reading the entire content. Extractive summarization
selects important sentences directly from the text, while abstractive summarization
generates new sentences that capture the core ideas.
"""

print("EXTRACTIVE SUMMARY (3 sentences)")
print("=" * 60)
print(extractive_summarize(text, num_sentences=3))

print("\nEXTRACTIVE SUMMARY (2 sentences)")
print("=" * 60)
print(extractive_summarize(text, num_sentences=2))

# ============================================================
# Program 5: Abstractive Summarization using BART (Transformers)
# ============================================================

from transformers import pipeline

# Load pre-trained BART summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# -- Example 1: COVID-19 -----------------------------------------------
text1 = """
The COVID-19 pandemic has caused unprecedented global disruption since its
emergence in late 2019. The virus, SARS-CoV-2, spread rapidly across the world,
leading to millions of deaths and widespread economic turmoil. Governments
worldwide implemented lockdowns, travel restrictions, and social distancing
measures to slow the virus's spread. Healthcare systems were stretched to their
limits as hospitals struggled to cope with the surge in patients. The pandemic
accelerated the development and rollout of vaccines at a record pace, with
multiple vaccines receiving emergency authorization within a year. Despite the
vaccines, new variants continued to emerge, posing challenges to global health.
The pandemic also highlighted existing health disparities and the importance of
global cooperation in addressing public health crises.
"""

print("EXAMPLE 1: COVID-19")
print("=" * 60)
print("Original Text:", text1[:200], "...")
summary1 = summarizer(text1, max_length=80, min_length=30, do_sample=False)
print("\nAbstractive Summary:")
print(summary1[0]['summary_text'])

# -- Example 2: Climate Change -----------------------------------------
text2 = """
Climate change refers to long-term shifts in global temperatures and weather
patterns. While some natural processes cause climate change, human activities
have been the main driver since the 1800s. The burning of fossil fuels such as
coal, oil, and gas generates greenhouse gas emissions that trap the sun's heat
and raise temperatures. Rising temperatures lead to more frequent and severe
weather events like hurricanes, floods, and droughts. Sea levels are rising
due to melting polar ice caps, threatening coastal communities. Scientists and
environmentalists are calling for urgent action to reduce carbon emissions and
transition to renewable energy sources to mitigate the worst effects.
"""

print("\nEXAMPLE 2: Climate Change")
print("=" * 60)
summary2 = summarizer(text2, max_length=70, min_length=25, do_sample=False)
print("Abstractive Summary:")
print(summary2[0]['summary_text'])

# ============================================================
# Program 6: Machine Translation using Helsinki-NLP Opus-MT
# ============================================================

from transformers import MarianMTModel, MarianTokenizer

def translate(text, src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer  = MarianTokenizer.from_pretrained(model_name)
    model      = MarianMTModel.from_pretrained(model_name)
    tokens     = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**tokens, num_beams=4, early_stopping=True)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


sentences = [
    "Natural Language Processing is a fascinating field of AI.",
    "Machine translation helps bridge the communication gap.",
    "Deep learning models can understand and generate human language.",
]

# -- English to French ---------------------------------------------------
print("English -> French")
print("=" * 60)
for s in sentences:
    print(f"EN: {s}")
    print(f"FR: {translate(s, 'en', 'fr')}")
    print("-" * 50)

# -- English to German ---------------------------------------------------
print("\nEnglish -> German")
print("=" * 60)
for s in sentences:
    print(f"EN: {s}")
    print(f"DE: {translate(s, 'en', 'de')}")
    print("-" * 50)

# -- English to Spanish -------------------------------------------------
print("\nEnglish -> Spanish")
print("=" * 60)
for s in sentences:
    print(f"EN: {s}")
    print(f"ES: {translate(s, 'en', 'es')}")
    print("-" * 50)

# ============================================================
# Program 7: Machine Translation using googletrans
# ============================================================

# Install: pip install googletrans==4.0.0-rc1
from googletrans import Translator, LANGUAGES

translator = Translator()

# -- Part A: Multi-Language Translation ----------------------------------
sentence = "Artificial Intelligence is changing the world."
target_langs = {
    'fr':    'French',
    'de':    'German',
    'es':    'Spanish',
    'hi':    'Hindi',
    'ta':    'Tamil',
    'ja':    'Japanese',
    'zh-cn': 'Chinese (Simplified)',
    'ar':    'Arabic',
}

print("PART A: Multi-Language Translation")
print("=" * 60)
print(f"Source (English): {sentence}\n")
for lang_code, lang_name in target_langs.items():
    result = translator.translate(sentence, dest=lang_code)
    print(f"{lang_name:25s}: {result.text}")

# -- Part B: Auto Language Detection + Translation to English -----------
print("\nPART B: Auto Language Detection + Translate to English")
print("=" * 60)
foreign_sentences = [
    "Bonjour, comment allez-vous?",       # French
    "Guten Morgen, wie geht es Ihnen?",   # German
    "Hola, como estas?",                  # Spanish
    "Namaste, aap kaise hain?",           # Hindi
    "Vanakkam, eppadi irukkireekal?",     # Tamil
]

for sent in foreign_sentences:
    detected   = translator.detect(sent)
    translated = translator.translate(sent, dest='en')
    lang_name  = LANGUAGES.get(detected.lang, detected.lang).capitalize()
    print(f"Input    : {sent}")
    print(f"Detected : {lang_name} (confidence: {detected.confidence:.2f})")
    print(f"English  : {translated.text}")
    print("-" * 50)
