# AI/ML Engineering: Learning Log

👋 Welcome to my public portfolio. This repository documents my structured, project-based journey to transition into AI Engineering, leveraging a fully cloud-native workflow.

## 🚀 Overview

- **My Goal:** To build a deep, practical understanding of machine learning by shipping production-ready, end-to-end AI systems.
- **My Approach:** A 4-month, intensive learning plan focused on building a sophisticated capstone project from the ground up.
- **My Stack:** This entire journey is executed on a **Chromebook**, utilizing a 100% cloud-based development environment (GitHub Codespaces, Google Colab, Hugging Face Spaces).

## 📅 Learning Plan: A Project-Based Roadmap

A high-level overview of my hands-on learning path. Each phase builds towards a comprehensive capstone project.

### **Phase 1: Foundations & Data Mastery**
- **Focus:** Mastering the data science stack (Pandas, NumPy, Seaborn) and core ML concepts through practical application.
- **Key Outcomes:** Proficiency in data cleaning, exploratory analysis, and building classical ML models.
- **Project:** [`API Data Ingestion`](learning_materials/week_2_data_manipulation) | [`Exploratory Data Analysis`](learning_materials/week_3_data_visualization/)

### **Phase 2: Deep Learning & NLP**
- **Focus:** Building and training neural networks with PyTorch and fine-tuning transformer models with Hugging Face.
- **Key Outcomes:** Ability to implement and fine-tune deep learning models for computer vision and NLP tasks.
- **Project:** [`Image Classifier`](learning_materials/week_5_fastai/) | [`NLP Text Classifier`](learning_materials/week_7_NLP/)

### **Phase 3: MLOps & System Integration**
- **Focus:** Transitioning from models to production systems. Building APIs with FastAPI, containerization with Docker, and implementing CI/CD.
- **Key Outcomes:** Skills in model deployment, containerization, and creating automated ML pipelines.
- **Project:** **Capstone: Support Ticket Triage Agent** (See below)

## 📂 Capstone Project: Intelligent Support Ticket Triage Agent

This end-to-end system is the central artifact of my learning journey, designed to demonstrate a full range of AI engineering skills.

| Component | Description | Tech Stack | Status |
| :--- | :--- | :--- | :--- |
| **1. Classification Model** | A fine-tuned Transformer model for classifying ticket urgency and category. | `PyTorch`, `Hugging Face Transformers` | `Complete` |
| **2. Prediction API** | A containerized FastAPI service that provides model predictions. | `FastAPI`, `Docker` | `Complete` |
| **3. CI/CD Pipeline** | Automated testing and deployment of the entire system. | `GitHub Actions` | `Complete` |
| **4. Live Demo** | The full system deployed and accessible. | `Hugging Face Spaces` | `Complete` |
| **5. Agentic Workflow** | An intelligent agent that decides on actions based on model output. | `MistralAI-powered agent` | `Complete` |

## 🛠️ Tech Stack

- **Development & CI/CD:** `GitHub Codespaces`, `GitHub Actions`
- **Experimentation & Training:** `Google Colab`, `Kaggle Notebooks`
- **Deployment & Demos:** `Hugging Face Spaces`, `Docker`
- **Frameworks & Libraries:** `PyTorch`, `Hugging Face Transformers`, `FastAPI`, `Scikit-learn`, `Pandas`, `NumPy`

## 📈 Progress Log

A living log of my weekly progress and key learnings.

# Progress Log

## Environment Setup
- [x] GitHub Codespaces configured with 2-core machine
- [x] Colab GPU access verified and tested
- [x] Basic requirements.txt created and packages installed

## Week 1 (Sep 8-14): Python Reactivation
- [x] Completed 10+ Exercism Python problems
- [x] Refreshed Python knowledge (Lists, Dictionaries, Functions, etc.)
- [x] Mastered list comprehensions and *args/**kwargs
- [x] Set up Black and Flake8 for code formatting

## Week 2 (Sep 15–21): Data Manipulation & Ingestion
- [x] Practiced Pandas basics on Titanic dataset (loading, cleaning, transformations)  
- [x] Created derived features (`FamilySize`, `AgeGroup`)  
- [x] Built API ingestion script with JSONPlaceholder and saved to DataFrame/CSV  
- [x] Applied `.groupby()`, `.agg()`, `.apply()` on Titanic & API datasets  

## Week 3 (Sep 22–28): Visualization & EDA
- [x] Polished Titanic EDA notebook (cleaning + insights)  
- [x] Created 3–5 visualizations (histogram, barplot, boxplot, heatmap) with detailed explanations  
- [x] Performed EDA on a second dataset (e.g., Iris) 
- [x] Practiced NumPy operations (arrays, reshaping, broadcasting)   

## Week 4 (Sep 29-Oct 5): ML Fundamentals
- [x] Implemented Logistic Regression on Iris dataset
- [x] Understood and explained gradient descent concepts

## Week 5 (Oct 6-12): Deep Learning Intro
- [x] Built Fast.ai image classifier
- [x] Learned CNN architecture fundamentals

## Week 6 (Oct 13-19): PyTorch Mastery
- [x] Created custom PyTorch training loop
- [x] Mastered backpropagation concepts

## Week 7 (Oct 20-26): NLP & Transformers
- [x] Implemented Hugging Face text classification pipeline
- [x] Learned transformer architecture components

## Week 8 (Oct 27-Nov 2): Capstone Classifier
- [x] Fine-tuned DistilBERT on support ticket data
- [x] Achieved F1 score > 0.85 on classification task

## Week 9 (Nov 3-9): API Development
- [x] Created FastAPI prediction endpoint
- [x] Implemented Docker containerization

## Week 10 (Nov 10-16): CI/CD Pipeline
- [x] Built GitHub Actions CI pipeline
- [x] Configured automated testing workflow

## Week 11 (Nov 17-23): Deployment
- [x] Deployed to Hugging Face Spaces
- [x] Verified health endpoints and functionality

## Week 12 (Nov 24-30): Agent Integration
- [x] Built Custom agent with business logic
- [x] Implemented Slack notification simulation

## Week 13 (Dec 1-7): Documentation
- [x] Created professional README with templates
- [x] Designed system architecture diagrams