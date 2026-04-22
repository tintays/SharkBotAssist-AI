# SharkBotAssist-AI
Project made by Josue Lopez Guevara for the course Computer Science 311: Artificial Intelligence.

---

## 1. Name and Purpose
The SharkBot-Assist AI is a technical support conversational agent developed to facilitate troubleshooting for the Shark IQ Robot (RV2100 Series). Its primary objective is to provide high-accuracy resolutions for hardware errors and maintenance routines by extracting information directly from official technical documentation, thereby eliminating the need for manual document navigation.

## 2. NLP/LLM Methods Used
This application utilizes a Retrieval-Augmented Generation (RAG) architecture to ensure factual grounding.
* **Core Engine:** Gemini 3 Flash (Google Generative AI).
* **Retrieval Mechanism:** Semantic Similarity Search using dense vector embeddings.
* **Contextual Constraint:** The agent operates within a "closed-domain" framework, meaning it is strictly instructed to answer queries based solely on the provided manual to mitigate model hallucinations.

## 3. Dataset Information
* **Source:** Official Shark IQ Robot Owner's Guide (PDF).
* **Data Format:** Unstructured technical documentation.
* **Records:** Approximately 56 semantic text segments.

### Data Engineering Pipeline
1. **Extraction:** Conversion of binary PDF data to normalized text, removing artifacts from multi-column layouts.
2. **Recursive Splitting:** Segmentation of text into 1000-character chunks to optimize the LLM’s context window.
3. **Contextual Overlap:** Implementation of a 150-character overlap to maintain semantic continuity for technical instructions spanning multiple segments.
4. **Embedding:** Generation of 384-dimensional dense vectors using the all-MiniLM-L6-v2 transformer.

## 4. Libraries, Toolkits, and Frameworks
* **Python 3.x:** Primary programming language.
* **LangChain:** Orchestration framework for RAG chain management.
* **FAISS (Facebook AI Similarity Search):** Vector database for efficient similarity retrieval.
* **HuggingFace:** Local embedding models for data processing and latency reduction.

## 5. Execution Instructions
To execute the chatbot in a local environment, follow these steps:

**1. Clone the repository:** `git clone https://github.com/tintays/SharkBotAssist-AI.git`  
`cd SharkBotAssist-AI`

**2. Install dependencies:** `pip install -r requirements.txt`

**3. Configure Environment Variables:** Set the Google AI Studio API Key:  
* Linux/Mac: `export GOOGLE_API_KEY="YOUR_KEY_HERE"`  
* Windows: `set GOOGLE_API_KEY="YOUR_KEY_HERE"`

**4. Run the Application:** `python SharkBotAssist_AI.py`

## 6. Results and Insights
The agent demonstrates significant precision in diagnostic tasks. Specifically, it successfully identifies that a "solid red light" indicates a brush motor obstruction or docking failure, providing the exact resolution steps found in the hardware guide. The implementation of a 150-character overlap was identified as a critical factor in maintaining the integrity of multi-step maintenance procedures.

---

## 9. References
* Raj Arun, R. (2024). *Mastering Large Language Models with Python*. Orange Education Pvt Ltd.
* Samaroo, A. (2025). *What is a Large Language Model (LLM)?*. Study.com.
* SharkNinja. (n.d.). *RV2100 Series User's Manual*.
