
# Elden Ring Story Chatbot

This repository contains code and notebooks for building and evaluating a chatbot that answers lore-based questions about *Elden Ring*. The bot is designed to produce rich, lore-consistent responses, using retrieval-augmented generation (RAG) to ground answers in specific context drawn from a custom-built database of lore and game details.

## Project Overview

The goal of this final project is to develop a RAG system or AI bot that combines the power of text and audio processing to answer questions about YouTube videos. The bot will utilize natural language processing (NLP) techniques and speech recognition to analyze both textual and audio input, extract relevant information from YouTube videos, and provide accurate answers to user queries.

The project comprises two main components:

1. **Final Bot Implementation** (`elden_bot.ipynb`): This notebook provides the complete process for creating the chatbot, including data preparation, model setup, and training with a RAG framework.
   
2. **Evaluation and Fine-Tuning** (`elden_bot_rag_evaluation.ipynb`): This notebook evaluates the chatbot’s effectiveness through three primary evaluation types: relevance, coherence, and engagement. Fine-tuning steps based on evaluation feedback help to refine the bot’s responses.

### Key Objectives

1. Develop a text-based question answering (QA) model using pre-trained language models. You may find it useful to fine-tune your model.
2. Integrate speech recognition capabilities to convert audio/video input (user questions) into text transcripts.
3. Build a conversational interface for users to interact with the bot via text or voice input. The latter is not a must.
4. Retrieve, analyze, and store into a vector database (chroma) YouTube video content to generate answers to user questions.
5. Test and evaluate the bot's performance in accurately answering questions about YouTube videos.
6. Your AI must use agents with several tools and memory.

## Getting Started

### Prerequisites

Install the following:

- Python 3.8 or later
- PyTorch
- Hugging Face Transformers
- Jupyter Notebook
- Additional dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/elden-ring-story-chatbot.git
   cd elden-ring-story-chatbot
   ```
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- **`elden_bot_final_version.ipynb`**: Notebook to preprocess data, configure the RAG architecture, train, and save the chatbot model.
- **`elden_bot_rag_evaluation.ipynb`**: Notebook to evaluate chatbot performance, covering three evaluation types for measuring quality.
- **`requirements.txt`**: Lists the packages required for running the project.
- **`data/`**: Folder to store lore data (optional, depending on your dataset structure).

## Training and Evaluating the Chatbot

### 1. Training the Chatbot

To build the chatbot with lore consistency:

1. **Open `elden_bot_final_version.ipynb`**:
   - Load lore data for *Elden Ring*, covering character stories, events, and world elements.
   - Preprocess text to prepare it for tokenization and model compatibility.

2. **Set Up the RAG Architecture**:
   - Configure the retriever and generator models.
   - Fine-tune the generator model to generate lore-specific responses using relevant passages fetched by the retriever.
   - Save the trained model.

### 2. Evaluating the Chatbot

The **`elden_bot_rag_evaluation.ipynb`** notebook assesses the chatbot using three primary evaluation types:

#### Evaluation Types

1. **Relevance**:
   - **Goal**: Ensure that responses directly answer the user’s question with pertinent information.
   - **Method**: Evaluate the alignment between user questions and the response’s content. This is typically done by manually assessing the response for context-appropriateness and accuracy.
   - **Example**: For a question like “Who is Malenia in *Elden Ring*?”, the bot’s answer should focus specifically on Malenia's role, background, and significance without veering off-topic.

2. **Coherence**:
   - **Goal**: Maintain a clear and logically structured response that flows naturally and is easy to understand.
   - **Method**: Analyze each response to verify that the answer has logical consistency, is grammatically correct, and lacks contradictions. This evaluation ensures that responses make sense within the lore framework and aren’t overly fragmented.
   - **Example**: When describing a character's background, the bot should avoid any inconsistencies in timeline or relationships, such as incorrectly merging multiple character storylines into one response.

3. **Engagement**:
   - **Goal**: Provide responses that are engaging and capture the tone and depth of the *Elden Ring* lore, making the interaction feel immersive.
   - **Method**: Assess the response’s ability to add value beyond basic facts by incorporating elements like detailed descriptions, thematic language, and storytelling. This evaluation helps determine if the bot can answer in a manner that feels as if it were part of the game’s world.
   - **Example**: For a question about the lore of the “Lands Between,” an engaging answer would describe the mythical elements and atmosphere of the location, rather than providing only geographic details.

### Example Code for Model Inference

Once the model is trained, here’s a basic example of how to interact with it:

```python
from transformers import AutoTokenizer, RagRetriever, RagTokenForGeneration

# Load tokenizer, retriever, and model
tokenizer = AutoTokenizer.from_pretrained("path/to/your/model")
retriever = RagRetriever.from_pretrained("path/to/your/retriever")
model = RagTokenForGeneration.from_pretrained("path/to/your/model")

# Ask a question
question = "What is the significance of the Elden Ring?"

# Encode and generate response
input_ids = tokenizer(question, return_tensors="pt").input_ids
generated = model.generate(input_ids, num_beams=2, max_length=100)
answer = tokenizer.decode(generated[0], skip_special_tokens=True)

print("Chatbot Answer:", answer)
```

## Future Improvements

- **Expand the Lore Database**: Incorporate a broader lore dataset to enhance response variety and depth.
- **Experiment with Alternative RAG Configurations**: Test different retriever-generator setups to optimize response coherence and engagement.
- **User Interface**: Develop a user-friendly frontend for easy interaction with the chatbot.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
