# Transformer-Based Sentence Embedding

## 1. Project Overview

This project is a simple Natural Language Processing application that converts a user-provided sentence into a numerical vector representation called a **sentence embedding**.

The project uses the pre-trained **all-MiniLM-L6-v2** Transformer model available through Hugging Face. Instead of training a neural network from the beginning, the project uses an already trained model to understand the input text and generate meaningful numerical representations.

The implementation is created using **Python, PyTorch, and Hugging Face Transformers** and developed in **Visual Studio Code**.

The project also contains a basic word-value mechanism that identifies selected words from the input sentence and displays their predefined numerical values.

---

## 2. Project Objective

The main objective of this project is to understand how modern Transformer-based models convert natural language into numerical representations.

The project demonstrates:

* Loading a pre-trained Transformer model
* Loading a compatible tokenizer
* Accepting text input from the user
* Converting text into tokens
* Creating PyTorch tensors from tokens
* Passing tokens through the Transformer model
* Extracting token-level embeddings
* Using an attention mask
* Applying mean pooling
* Creating a sentence-level embedding
* Converting tensors into NumPy arrays
* Displaying the generated embedding
* Matching predefined words with numerical values

---

## 3. Technologies Used

| Technology                | Purpose                                 |
| ------------------------- | --------------------------------------- |
| Python                    | Main programming language               |
| PyTorch                   | Tensor operations and model execution   |
| Hugging Face Transformers | Loading tokenizer and Transformer model |
| Hugging Face Model Hub    | Provides the pre-trained model          |
| Visual Studio Code        | Development environment                 |

---

## 4. Model Used

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model is designed to generate useful representations of sentences and short text.

The model is loaded using:

```python
model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

When the program is executed for the first time, the required model files are downloaded automatically.

---

## 5. Project Architecture

The overall workflow of the project is:

```text
User Input
    |
    v
Text Tokenization
    |
    v
PyTorch Tensor Conversion
    |
    v
Transformer Model
    |
    v
Token-Level Embeddings
    |
    v
Attention Mask
    |
    v
Mean Pooling
    |
    v
Sentence Embedding
    |
    v
Numerical Vector
```

A separate word-value process is also performed:

```text
Input Sentence
      |
      v
Convert to Lowercase
      |
      v
Split into Words
      |
      v
Compare with Dictionary
      |
      v
Display Matching Word Values
```

---

## 6. Project Files

```text
Embedding model/
│
├── embedding.py
├── requirements.txt
└── README.md
```

### embedding.py

Contains the Python implementation for generating sentence embeddings.

### requirements.txt

Contains the required Python packages.

### README.md

Contains project information and execution instructions.

---

## 7. Importing Libraries

The project begins with:

```python
from transformers import AutoTokenizer, AutoModel
import torch
```

### AutoTokenizer

`AutoTokenizer` converts human-readable text into tokens that can be understood by the Transformer model.

### AutoModel

`AutoModel` loads the neural network architecture and pre-trained weights.

### PyTorch

PyTorch is used to perform tensor operations and execute the model.

---

## 8. Loading the Pre-trained Model

The model name is stored in a variable:

```python
model_name = "sentence-transformers/all-MiniLM-L6-v2"
```

The tokenizer is loaded:

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

The model is loaded:

```python
model = AutoModel.from_pretrained(model_name)
```

This allows the program to use the knowledge learned by the pre-trained Transformer model.

---

## 9. Getting User Input

The program accepts a sentence from the user:

```python
sentence = input("Enter a sentence: ")
```

For example:

```text
Enter a sentence: I enjoy programming
```

The entered sentence is stored in the `sentence` variable.

---

## 10. Tokenization

The input sentence is passed to the tokenizer:

```python
inputs = tokenizer(
    sentence,
    return_tensors="pt",
    padding=True,
    truncation=True
)
```

Tokenization converts the sentence into a format that can be processed by the Transformer model.

### return_tensors

```python
return_tensors="pt"
```

This tells the tokenizer to return PyTorch tensors.

### padding

```python
padding=True
```

Padding ensures that inputs can have compatible dimensions when necessary.

### truncation

```python
truncation=True
```

Truncation prevents the input from exceeding the model's supported sequence length.

---

## 11. Generating Model Output

The tokenized input is passed to the Transformer model:

```python
with torch.no_grad():
    outputs = model(**inputs)
```

`torch.no_grad()` disables gradient calculation because this project only performs inference.

The model produces contextual representations for the input tokens.

---

## 12. Extracting Token Embeddings

The final hidden state is obtained using:

```python
token_embeddings = outputs.last_hidden_state
```

The `last_hidden_state` contains a vector representation for every token in the input sentence.

Therefore, the output contains multiple token embeddings rather than one single sentence embedding.

---

## 13. Attention Mask

The attention mask is extracted:

```python
attention_mask = inputs["attention_mask"]
```

The attention mask indicates which positions contain actual tokens.

It helps prevent padding tokens from affecting the final sentence representation.

---

## 14. Creating the Mask

The mask is expanded to match the dimensions of the token embeddings:

```python
mask = attention_mask.unsqueeze(-1).expand(
    token_embeddings.size()
).float()
```

This allows the mask to be multiplied with the token embeddings.

---

## 15. Calculating Sum of Embeddings

The token embeddings are multiplied by the mask:

```python
sum_embeddings = torch.sum(
    token_embeddings * mask,
    dim=1
)
```

This adds the valid token representations while ignoring masked positions.

---

## 16. Calculating Valid Token Count

The project calculates the number of valid tokens:

```python
sum_mask = torch.clamp(
    mask.sum(dim=1),
    min=1e-9
)
```

The `torch.clamp()` function prevents the denominator from becoming zero.

---

## 17. Mean Pooling

The sentence embedding is generated using mean pooling:

```python
embedding = sum_embeddings / sum_mask
```

Mean pooling calculates the average of the valid token embeddings.

The resulting vector provides a single numerical representation for the complete sentence.

---

## 18. Displaying the Sentence

The input sentence is displayed:

```python
print("\nSentence:")
print(sentence)
```

Example:

```text
Sentence:
I enjoy programming
```

---

## 19. Displaying the Sentence Embedding

The embedding is displayed using:

```python
print("\nEmbedding:")
print(embedding[0].numpy())
```

The tensor is converted into a NumPy array before displaying the values.

The output consists of numerical values representing the sentence.

---

## 20. Word Value Mapping

The project contains a simple dictionary:

```python
word_values = {
    "coding": 1,
    "programming": 1,
    "cooking": 2,
    "music": 3
}
```

The dictionary assigns a numerical value to selected words.

| Word        | Assigned Value |
| ----------- | -------------: |
| coding      |              1 |
| programming |              1 |
| cooking     |              2 |
| music       |              3 |

---

## 21. Word Matching

The program checks every word in the input:

```python
for word in sentence.lower().split():
    if word in word_values:
        print(word, ":", word_values[word])
```

First, the sentence is converted into lowercase.

Then it is divided into individual words.

Each word is compared with the `word_values` dictionary.

If a matching word is found, its value is displayed.

---

## 22. Example

### Input

```text
Enter a sentence: I love coding and music
```

### Output

```text
Sentence:
I love coding and music

Embedding:
[ ... numerical values ... ]

Word Values:
coding : 1
music : 3
```

The exact embedding values depend on the Transformer model and the input sentence.

---

## 23. Installation

Make sure Python 3.12 is installed.

Check the Python version:

```bash
python --version
```

Then install the required libraries:

```bash
python -m pip install transformers torch
```

Alternatively, install dependencies using:

```bash
python -m pip install -r requirements.txt
```

---

## 24. requirements.txt

Create a file named:

```text
requirements.txt
```

Add:

```text
transformers
torch
```

---

## 25. Running the Project

Open the project folder in **Visual Studio Code**.

Open:

```text
Terminal → New Terminal
```

Run:

```bash
python embedding.py
```

The program will ask:

```text
Enter a sentence:
```

Enter any sentence and press Enter.

---

## 26. Sample Run

```text
Enter a sentence: I love coding

Sentence:
I love coding

Embedding:
[ 0.012 ... -0.034 ... 0.056 ... ]

Word Values:
coding : 1
```

---

## 27. Key Concepts Learned

This project helps understand several important NLP concepts.

### Tokenization

Converting text into tokens that a Transformer can process.

### Transformer

A neural network architecture designed to process sequential data and understand contextual relationships between tokens.

### Token Embedding

A numerical representation generated for each token.

### Attention Mask

A mechanism used to identify valid tokens and ignore padding.

### Mean Pooling

A technique for combining token embeddings into one sentence-level vector.

### Sentence Embedding

A numerical representation of the meaning or semantic information contained in a sentence.

---

## 28. Applications

The generated sentence embeddings can be used as a foundation for:

* Semantic Search
* Text Similarity
* Document Matching
* Recommendation Systems
* Question Answering
* Chatbots
* Information Retrieval
* Text Classification
* Duplicate Text Detection
* Document Retrieval
* Retrieval-Augmented Generation
* Natural Language Processing applications

---

## 29. Advantages

* Uses a pre-trained Transformer model.
* Does not require training a model from scratch.
* Converts natural language into numerical vectors.
* Uses contextual information from the Transformer.
* Simple Python implementation.
* Easy to modify and extend.
* Can be used as a foundation for advanced NLP applications.

---

## 30. Limitations

* The first execution requires downloading the model.
* Internet access is required for the initial model download.
* The model requires system memory to load.
* The word-value system only recognizes predefined words.
* The word matching logic does not perform semantic matching.
* Punctuation and different word forms may affect the simple dictionary matching.

---

## 31. Future Improvements

The project can be extended with:

1. Cosine similarity between two sentences.
2. Similarity score calculation.
3. Semantic search.
4. Multiple sentence comparison.
5. Document embeddings.
6. Vector database integration.
7. Streamlit user interface.
8. RAG implementation.
9. Text classification.
10. Chatbot integration.

---

## 32. Development Environment

This project was developed using:

```text
Python 3.12
Visual Studio Code
PyTorch
Hugging Face Transformers
```

The implementation was written and executed in **Visual Studio Code**.

---

## 33. Conclusion

This project demonstrates how a pre-trained Transformer model can be used to convert a natural language sentence into a numerical embedding.

The implementation covers the complete basic workflow from user input and tokenization to Transformer processing, attention-mask handling, mean pooling, and final sentence embedding generation.

The additional word-value system demonstrates basic dictionary-based word identification.

Overall, this project provides a practical introduction to Transformer models, sentence embeddings, and fundamental Natural Language Processing concepts.

## Author

Keerthana
