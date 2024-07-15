from transformers import pipeline

# tests2
# Load the pre-trained sentiment analysis model
sentiment_analysis = pipeline(
"sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

input_text = [
"It’s a great app, there are no problem with this app and has a great user interface. I am very happy with the customer service! This increased our efficiency."
]

# Perform sentiment analysis on the input text
result = sentiment_analysis(input_text)

# Print the result
print(result)
