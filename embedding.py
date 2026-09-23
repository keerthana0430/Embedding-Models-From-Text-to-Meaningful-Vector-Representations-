from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")


sentence = input("Enter a sentence: ")


embedding = model.encode(sentence)

print("\nSentence:")
print(sentence)

print("\nEmbedding:")
print(embedding)


word_values = {
    "coding": 1,
    "programming": 1,
    "cooking": 2,
    "music": 3
}


example_sentence = "I enjoy coding"

print("\nWord Values:")

words = example_sentence.lower().split()

for word in words:
    if word in word_values:
        print(word, ":", word_values[word])
