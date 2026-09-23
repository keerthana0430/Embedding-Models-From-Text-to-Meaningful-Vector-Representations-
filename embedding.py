from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get sentence from user
sentence = input("Enter a sentence: ")

# Generate embedding
embedding = model.encode(sentence)

print("\nSentence:")
print(sentence)

print("\nEmbedding:")
print(embedding)

# Word values
word_values = {
    "coding": 1,
    "programming": 1,
    "cooking": 2,
    "music": 3
}

# Example sentence
example_sentence = "I enjoy coding"

print("\nWord Values:")

words = example_sentence.lower().split()

for word in words:
    if word in word_values:
        print(word, ":", word_values[word])