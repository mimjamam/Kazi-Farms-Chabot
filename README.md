# Kazi Farms Chatbot

A simple chatbot for Kazi Farms employee queries using Streamlit and Groq.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

3. Run the app:
   ```bash
   streamlit run main.py
   ```

## Features

- Employee policy and salary queries
- Auto-cleanup memory on tab close
- Personal information protection
- Session timeout after 30 minutes
