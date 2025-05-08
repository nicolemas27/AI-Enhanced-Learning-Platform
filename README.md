# AI Enhanced Learning Platform 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

This platform transforms YouTube videos into interactive and comprehensive learning experiences using advanced AI. It generates **quizzes**, **summaries**, **flashcards**, and **concept maps**, adapts to individual learning styles, and tracks progress to optimize retention with personalized learning schedules.


## Features

### Core Tools
- **Adaptive Quiz Engine**: Implements difficulty adjustment algorithms that analyze user performance data to generate progressively challenging questions
- **Intelligent Summarization**: Utilizes NLP techniques to extract key concepts and produce hierarchical summaries
- **Interactive Flashcards**: Generates study cards with semantic relevance sorting
- **Knowledge Graph Visualization**: Creates concept maps using graph theory algorithms to display relationships between key concepts
- **Multilingual Support**: Translates learning content into multiple languages

### Adaptive Learning

- **Memory Retention Models**: Uses spaced repetition for optimal learning
- **Personalized Difficulty Adjustment**: Adapts content based on performance
- **Weak Concept Identification**: Highlights areas needing more focus
- **Performance Analytics Dashboard**: Visualizes user progress metrics and learning analytics
- **Enhanced Memory Model**: Custom implementation of forgetting curve algorithms with personalized decay rates
- **Multi-model Concept Detection**: Combines TF-IDF, spaCy NER, and LLM for robust concept extraction

## Technologies

| Component         | Technologies Used                                    |
|-------------------|----------------------------------------------------- |
| Backend           | Python, Flask, Flask-Login                           |
| Database          | MongoDB Atlas (PyMongo)                              |
| AI/ML             | LLM, scikit-learn, NumPy, Custom Memory Models       |
| Data Processing   | BeautifulSoup, pytube, youtube_transcript_api        |
| Frontend          | Bootstrap, D3.js, Chart.js, HTML/CSS/JavaScript      |

## Setup

1. **Clone repository**
   ```bash
   git clone https://github.com/nicolemas27/AI-Enhanced-Learning-Platform.git
   cd AI-Enhanced-Learning-Platform 

2. **Create virtual environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

4. **MongoDB Setup**
   1. Create a MongoDB Atlas account or use a local MongoDB installation
   2. Add your connection string to the .env file
   
5. **Configure environment**
   ```bash
   echo "MONGO_URI=your_mongodb_uri" > .env
   echo "GEMINI_KEY=your_api_key" >> .env
   echo "SECRET_KEY=your_secret_key" >> .env

6. **Run the Application**
   ```bash
   python app.py

## 🧩 Usage

### Home Page
- Enter a **YouTube URL** and select the desired learning tool.

### Learning Tools
- **Quiz**: Test your knowledge with adaptive questions.
- **Flashcards**: Use interactive cards for efficient memorization.
- **Summary**: Review concise explanations of key concepts.
- **Knowledge Graph**: Explore visual relationships between concepts.
- **Progress Dashboard**: Track your learning metrics and personalized review schedule.


## 🧩 Chrome Extension

The platform also includes a **Chrome Extension** for seamless integration with YouTube:

1. Navigate to the `Extension-Quiz` directory in the project.
2. Open **Chrome** and go to `chrome://extensions/`.
3. Enable **Developer mode** (top right).
4. Click **"Load unpacked"** and select the `Extension-Quiz` folder.
5. Use the extension while watching YouTube videos to instantly access the learning tools.


   
