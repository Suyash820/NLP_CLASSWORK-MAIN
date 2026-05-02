from transformers import pipeline

# Load pretrained model
classifier = pipeline("sentiment-analysis")

# Input
text = "I love using BERT models!"

# Output
result = classifier(text)

print("Input:", text)
print("Output (BERT prediction):", result)