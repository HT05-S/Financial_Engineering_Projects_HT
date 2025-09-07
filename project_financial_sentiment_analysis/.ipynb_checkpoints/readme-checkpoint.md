## Project Report: Financial Sentiment Analyzer

## 1. Project Overview-
This report details the development of a Financial Sentiment Analyzer, a specialized Natural Language Processing (NLP) application designed to classify financial news sentiment. The project leverages a pre-trained FinBERT model, which has been fine-tuned to recognize positive, negative, and neutral sentiments within financial contexts. The core deliverable is a Streamlit web application that provides a user-friendly interface for two modes of sentiment analysis: manual text input and automated processing of real-time news data from the Alpha Vantage API. This project demonstrates a practical, end-to-end implementation of an NLP solution for financial market analysis.

## 2. Project Lifecycle & Deliverables-

![alt text](image.png)

A key technical result from the fine-tuning process was the determination of an optimal softmax temperature of 1.1315. This parameter was crucial in calibrating the model's confidence scores, yielding more reliable probability distributions for its predictions.

## 3. Results & Discussion-
- The project successfully created a functional and practical tool for financial sentiment analysis. Key results include:

- Robust Sentiment Classification: The fine-tuned FinBERT model effectively categorizes financial news, providing nuanced sentiment insights (positive, negative, neutral) that are essential for market participants.

- Actionable Intelligence: By combining the sentiment model with real-time news from Alpha Vantage, the application offers immediate, data-driven insights into market sentiment, which can be a valuable input for trading strategies or risk assessment.

- Accessibility: The Streamlit application democratizes access to this advanced NLP capability, presenting complex model outputs in a clear, tabular format that is easy for non-technical users to interpret.

## 4. Risks-
- Data Dependencies: The application's functionality is contingent on the availability and reliability of the Alpha Vantage API. Rate limits and potential service downtime pose a risk to continuous operation.

- Model Generalization: The model’s performance is optimized for the language within its training data. It may not generalize as effectively to highly idiosyncratic or novel financial terminology.

- Market Dynamics: The financial landscape is constantly evolving. The model’s performance may degrade over time as market vernacular and news reporting styles change.

## 5. Assumptions-
- Input Data Quality: It was assumed that the news data retrieved from the Alpha Vantage API is clean and representative of genuine financial reporting.

- Model Loading: The project assumes that the fine-tuned model files (config.json, pytorch_model.bin, etc.) are portable and can be loaded correctly into the Streamlit environment without pathing or versioning issues.