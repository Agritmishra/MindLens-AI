# 🧠 MindLens: AI-Powered Reflection Companion

**Empathetic AI that listens, understands emotions.**

---

## Overview

**MindLens** is an emotional insight engine and journaling companion built using **Hugging Face Transformers**, **Streamlit**, and modern AI technologies.
It analyzes personal reflections, detects emotional tone, summarizes thoughts, and suggests personalized **reflection prompts** and **micro-actions** for emotional growth.

---

## ✨ Key Features

| Category                           | Description                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------- |
|  **Emotional Insight Engine**    | Analyzes text input using fine-tuned Hugging Face models (or zero-shot fallback).           |
|  **Smart Summarization**         | Extracts meaningful summaries from user reflections for cognitive clarity.                  |
|  **Adaptive Prompts**            | Suggests reflection questions and micro-actions tailored to emotional state.                |
|  **Privacy-First Design**        | All processing is local — no data leaves your system.                                       |
|  **Explainable & Modular**       | Core logic in `insight_engine.py` is easily extendable and interpretable.                   |
|  **Lightweight Heuristics**      | Works even without large models — with graceful fallback for offline use.                   |
|  **Emotion-Aware Micro-Actions** | Offers tiny, mindful behavioral suggestions to support mental balance.                      |
|  **Tech Integration**            | Uses Python, Streamlit, Hugging Face, Transformers, PyTorch, NumPy, Matplotlib, and dotenv. |
|  **Extensible Framework**        | Modular design supports custom model fine-tuning and multimodal expansion.                  |

---

## System Architecture

### High-Level Flow

```
User Input (Text)
     │
     ▼
Streamlit UI (app.py)
     │ Handles text input, output rendering, and visualization.
     ▼
Insight Engine (insight_engine.py)
     │ Responsible for emotion detection, summarization, and insights.
     ├── Model Handler Layer
     │     ├── Fine-tuned classifier (text classification)
     │     ├── Zero-shot classifier fallback (MNLI-based)
     │     └── Summarizer (optional)
     │
     ├── Heuristic Engine
     │     ├── Keyword emotion mapping (positive/negative lexicon)
     │     ├── Mood confidence scoring
     │     └── Balance computation (positivity index)
     │
     ├── Reflection Generator
     │     ├── Contextual question prompts
     │     └── Actionable micro-tasks
     │
     └── Output Composer
           ├── Summary
           ├── Mood and confidence score
           ├── Reflection prompts & micro-actions
           └── Well-being suggestions

     ▼
Streamlit UI Output Layer
     └── Displays insights, visual mood scores, and balanced reflection cards.
```

### Component Overview

| Component                       | Description                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------- |
| **UI Layer (Streamlit)**        | Provides interactive form, handles inputs, displays analytics and summaries.  |
| **Core Engine (InsightEngine)** | Main NLP and reasoning unit that drives analysis and mood detection.          |
| **Model Subsystem**             | Integrates with Hugging Face pipelines for classifier and summarizer.         |
| **Heuristic Layer**             | Lightweight backup when no models are available — ensures continuity offline. |
| **Data Flow Layer**             | Maintains smooth interaction between UI and engine using cached resources.    |
| **Micro-Action System**         | Curates small, emotion-based behavioral suggestions for reflection.           |
| **Security Layer**              | Uses dotenv to load and protect private Hugging Face tokens.                  |

---

## 🧠 Technologies Used

* **Python 3.10+** – Core language
* **Streamlit** – Interactive UI framework
* **Hugging Face Transformers** – Pretrained NLP pipelines
* **PyTorch** – Model backend
* **dotenv** – Secure Hugging Face token handling
* **Matplotlib & NumPy** – Lightweight analytics
* **Hugging Face Hub** – Model hosting
* **Regular Expressions (re)** – Text parsing and emotion extraction
* **Random / Counter modules** – Sampling and lexical analysis support

---


## 🧘 Example Output

**Input:**

> “I’ve been feeling anxious about my work lately, but I’m trying to stay hopeful.”

**Output:**

* **Mood:** Anxious
* **Confidence:** 83%
* **Summary:** You’re reflecting on anxiety while maintaining hope.
* **Reflection Prompts:**

  * What outcome are you trying too hard to control?
  * Which past anxiety turned out easier than expected?
* **Micro Actions:**

  * Practice box-breathing (4-4-4-4) for one minute.
  * Write one reassuring truth from your past experience.

---

## 🧭 Research & Academic Significance

* **Human–AI Emotional Interaction:** Demonstrates affective modeling and empathy simulation.
* **AI Ethics:** Prioritizes privacy and interpretability — no cloud logging or data retention.
* **Cognitive Computing:** Bridges AI text understanding with psychology and mindfulness.
* **Scalability:** Modular design supports multimodal input (text, voice, image) and local deployment.

---

## File Structure

```
├── app.py               # Streamlit front-end
├── insight_engine.py    # AI logic core
├── requirements.txt     # Dependencies
└── .env                 # Private Hugging Face token
```

## 🧾 License

Open for educational and non-commercial research use.

