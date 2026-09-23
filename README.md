# Sentence Embedding and Word Value Model

## Project Description

This project demonstrates how to generate a numerical representation of a sentence using a pre-trained Transformer model.

The project uses the `sentence-transformers/all-MiniLM-L6-v2` model from Hugging Face. The sentence entered by the user is converted into tokens using the tokenizer, processed by the Transformer model, and converted into a fixed-size embedding using mean pooling.

The project also includes a simple word-value system. Specific words such as `coding`, `programming`, `cooking`, and `music` are assigned numerical values. If any of these words are present in the entered sentence, their corresponding values are displayed.

This project provides a basic understanding of how text can be converted into numerical vectors and how Transformer-based models can be used for Natural Language Processing tasks.

## Objectives

The main objectives of this project are:

* To understand sentence embeddings.
* To understand how Transformer models process text.
* To convert a sentence into tokens.
* To generate numerical representations of text.
* To understand attention masks.
* To implement mean pooling.
* To generate a sentence embedding vector.
* To understand how embeddings can represent the meaning of text.
* To identify specific words from a sentence.
* To assign numerical values to selected words.
* To understand the basic workflow of a Transformer-based NLP application.

## Technologies Used

* Python 3.12
* PyTorch
* Hugging Face Transformers
* Hugging Face Model Hub
* Visual Studio Code

## Libraries Used

The project uses the following Python libraries:

```text
transformers
torch
```

### Transformers

The Hugging Face Transformers library is used to load the pre-trained tokenizer and Transformer model.

The following classes are used:

```python
from transformers import AutoTokenizer, AutoModel
```

### PyTorch

PyTorch is used for tensor operations, model execution, attention-mask processing, and mean pooling.

```python
import torch
```

## Pre-trained Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This is a pre-trained Transformer model designed for generating sentence embeddings.

The model is loaded using:

```python
model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

The first time the program is executed, the required model files are downloaded automatically from Hugging Face.

## Project Structure

```text
Embedding model/
│
├── embedding.py
├── requirements.txt
└── README.md
```

### embedding.py

Contains the complete Python implementation for:

* Loading the Transformer model
* Tokenizing the input sentence
* Generating model outputs
* Applying mean pooling
* Creating the sentence embedding
* Displaying the embedding
* Finding predefined words
* Displaying word values

### requirements.txt

Contains the Python dependencies required to run the project.

### README.md

Contains the project description, installation instructions, usage instructions, and technical explanation.

## How the Project Works

The project follows these main steps:

```text
User enters a sentence
          ↓
Sentence is tokenized
          ↓
Tokens are converted into tensors
          ↓
Transformer model processes the tokens
          ↓
Token embeddings are generated
          ↓
Attention mask is applied
          ↓
Mean pooling is performed
          ↓
Sentence embedding is generated
          ↓
Embedding is displayed
          ↓
Predefined words are checked
          ↓
Word values are displayed
```

## Step 1: Import Required Libraries

The project begins by importing the required libraries:

```python
from transformers import AutoTokenizer, AutoModel
import torch
```

`AutoTokenizer` is used to convert the input sentence into tokens.

`AutoModel` is used to load the pre-trained Transformer model.

`torch` is used for tensor operations and model execution.

## Step 2: Load the Model

The model name is specified as:

```python
model_name = "sentence-transformers/all-MiniLM-L6-v2"
```

The tokenizer is loaded using:

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

The Transformer model is loaded using:

```python
model = AutoModel.from_pretrained(model_name)
```

The tokenizer and model work together to process the input sentence.

## Step 3: Get User Input

The program asks the user to enter a sentence:

```python
sentence = input("Enter a sentence: ")
```

For example:

```text
Enter a sentence: I love programming
```

The entered sentence is stored in the `sentence` variable.

## Step 4: Tokenization

The sentence is converted into tokens using:

```python
inputs = tokenizer(
    sentence,
    return_tensors="pt",
    padding=True,
    truncation=True
)
```

### Parameters

#### return_tensors="pt"

This converts the tokenized input into PyTorch tensors.

#### padding=True

This adds padding when required so that inputs have compatible lengths.

#### truncation=True

This ensures that very long input text is truncated according to the model's maximum input length.

## Step 5: Generate Embeddings

The model processes the tokenized input:

```python
with torch.no_grad():
    outputs = model(**inputs)
```

`torch.no_grad()` is used because this project only needs the model output and does not need to calculate gradients.

This reduces unnecessary memory usage during inference.

## Step 6: Get Token Embeddings

The last hidden state is obtained using:

```python
token_embeddings = outputs.last_hidden_state
```

The last hidden state contains numerical representations for the tokens in the input sentence.

Each token has its own vector representation.

## Step 7: Get Attention Mask

The attention mask is obtained using:

```python
attention_mask = inputs["attention_mask"]
```

The attention mask indicates which tokens are actual input tokens and which positions are padding.

This is important because padding should not contribute to the final sentence embedding.

## Step 8: Prepare the Attention Mask

The attention mask is expanded:

```python
mask = attention_mask.unsqueeze(-1).expand(
    token_embeddings.size()
).float()
```

This makes the attention mask compatible with the dimensions of the token embeddings.

## Step 9: Sum the Token Embeddings

The token embeddings are multiplied by the attention mask:

```python
sum_embeddings = torch.sum(
    token_embeddings * mask,
    dim=1
)
```

This calculates the sum of the valid token embeddings while ignoring padding tokens.

## Step 10: Calculate the Number of Valid Tokens

The project calculates the total number of valid tokens:

```python
sum_mask = torch.clamp(
    mask.sum(dim=1),
    min=1e-9
)
```

The minimum value prevents division by zero.

## Step 11: Mean Pooling

Mean pooling is performed using:

```python
embedding = sum_embeddings / sum_mask
```

Mean pooling combines the individual token embeddings into a single vector representing the complete sentence.

This is an important part of the project because the Transformer produces an embedding for each token, while the application needs one embedding representing the whole sentence.

## Step 12: Display the Sentence

The entered sentence is displayed:

```python
print("\nSentence:")
print(sentence)
```

Example:

```text
Sentence:
I love programming
```

## Step 13: Display the Embedding

The generated sentence embedding is displayed using:

```python
print("\nEmbedding:")
print(embedding[0].numpy())
```

The output is a numerical vector.

Example:

```text
Embedding:
[ 0.0123 -0.0456  0.0789 ... ]
```

The actual values depend on the input sentence and the model.

## Word Value System

The project also contains a simple dictionary of predefined word values:

```python
word_values = {
    "coding": 1,
    "programming": 1,
    "cooking": 2,
    "music": 3
}
```

Each word has a numerical value.

| Word        | Value |
| ----------- | ----: |
| coding      |     1 |
| programming |     1 |
| cooking     |     2 |
| music       |     3 |

## Finding Word Values

The program checks each word in the entered sentence:

```python
for word in sentence.lower().split():
    if word in word_values:
        print(word, ":", word_values[word])
```

The sentence is converted to lowercase using:

```python
sentence.lower()
```

Then the sentence is divided into individual words using:

```python
split()
```

If a word exists in the `word_values` dictionary, its value is displayed.

## Example Input

```text
Enter a sentence: I love coding and programming
```

The program generates the sentence embedding.

It also checks the words against the predefined dictionary.

Example:

```text
Word Values:
coding : 1
programming : 1
```

Another example:

```text
Enter a sentence: I enjoy cooking and music
```

Output:

```text
Word Values:
cooking : 2
music : 3
```

## Installation

### Step 1: Install Python

Install Python 3.12 on the system.

Check the Python version:

```bash
python --version
```

Expected output:

```text
Python 3.12.x
```

### Step 2: Open the Project in VS Code

Open the project folder in Visual Studio Code.

```text
Embedding model
```

### Step 3: Open VS Code Terminal

In VS Code:

```text
Terminal → New Terminal
```

Make sure the terminal is opened inside the project folder.

### Step 4: Install Dependencies

Run:

```bash
python -m pip install -r requirements.txt
```

If `requirements.txt` is not available, install the libraries directly:

```bash
python -m pip install transformers torch
```

## requirements.txt

The `requirements.txt` file can contain:

```text
transformers
torch
```

## How to Run

Open the VS Code terminal inside the project folder.

Run:

```bash
python embedding.py
```

The program will display:

```text
Enter a sentence:
```

Enter any sentence and press Enter.

For example:

```text
Enter a sentence: Artificial Intelligence is useful
```

The program will generate the embedding and display the sentence.

## Sample Execution

```text
Enter a sentence: I love coding

Sentence:
I love coding

Embedding:
[ ... numerical vector values ... ]

Word Values:
coding : 1
```

## Important Concepts

### Sentence Embedding

A sentence embedding is a numerical representation of a sentence.

It allows text to be represented as vectors that can be processed mathematically.

### Tokenization

Tokenization converts text into smaller units called tokens.

For example:

```text
I love coding
```

can be divided into token representations before being processed by the Transformer.

### Transformer Model

A Transformer model processes the input tokens and generates contextual representations.

### Attention Mask

An attention mask identifies valid tokens and prevents padding tokens from affecting the embedding calculation.

### Mean Pooling

Mean pooling calculates the average representation of the valid token embeddings.

The resulting vector represents the sentence as a whole.

## Applications

Sentence embeddings can be used in many Natural Language Processing applications, including:

* Semantic Search
* Text Similarity
* Document Similarity
* Recommendation Systems
* Question Answering
* Information Retrieval
* Chatbots
* Natural Language Processing
* Retrieval-Augmented Generation
* Text Classification
* Duplicate Question Detection

## Advantages

* Uses a pre-trained Transformer model.
* No need to train a model from scratch.
* Converts sentences into numerical vectors.
* Can process different types of text.
* Useful for semantic text processing.
* Easy to implement using Python.
* Can be extended to larger NLP applications.

## Limitations

* The model needs to be downloaded before the first use.
* Large models may require more memory.
* The quality of the embedding depends on the selected pre-trained model.
* The simple word-value system only recognizes words explicitly included in the dictionary.
* Punctuation can affect the simple word matching logic.

## Future Enhancements

The project can be extended by adding:

* Cosine similarity between two sentences.
* Semantic search.
* Multiple sentence comparison.
* Text classification.
* A graphical user interface.
* Streamlit web interface.
* Document embedding.
* Vector database integration.
* Retrieval-Augmented Generation.
* Similarity score calculation.
* Support for larger Transformer models.

## Development Environment

This project was developed using:

```text
Python
Visual Studio Code
PyTorch
Hugging Face Transformers
```

The project was created and tested in **Visual Studio Code (VS Code)**.

## Author

Keerthana

## Conclusion

This project demonstrates the basic process of converting human language into numerical representations using a pre-trained Transformer model.

The project covers important concepts such as tokenization, Transformer models, attention masks, token embeddings, mean pooling, and sentence embeddings.

It also demonstrates a simple word-value matching system that identifies selected words from the user's input sentence.

This project provides a foundation for developing more advanced Natural Language Processing applications such as semantic search, text similarity systems, recommendation systems, chatbots, and Retrieval-Augmented Generation systems.
