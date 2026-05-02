from transformers import pipeline

# Load RoBERTa sentiment model
classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

# Input
text = "I do not like this product at all, it's terrible!"

# Output
result = classifier(text)

print("Input:", text)
print("Output (RoBERTa prediction):", result)