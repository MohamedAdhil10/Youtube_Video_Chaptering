# 📺 YouTube Video Chaptering using Transcript and Topic Modeling

This project uses the YouTube Transcript API and Natural Language Processing techniques to automatically segment a YouTube video into chapters based on the transcript content.

## 🚀 Features

- Extracts transcripts from YouTube videos using the `youtube_transcript_api`
- Uses the YouTube Data API to retrieve video metadata (title)
- Analyzes transcript segments for length and word frequency
- Applies **Topic Modeling** using **NMF (Non-negative Matrix Factorization)** to identify key themes
- Detects logical breaks between segments to define video chapters
- Extracts and labels chapter names using **TF-IDF**

## 🧠 Technologies Used

- Python (Google Colab)
- YouTube Data API (v3)
- `youtube_transcript_api`
- `scikit-learn`
- `matplotlib`, `pandas`, `numpy`
- NLP techniques: CountVectorizer, TF-IDF, Topic Modeling (NMF)

## ⚠️ Important Note

❗ **Do NOT include your API key in public repositories.**  
Instead, use environment variables or a `.env` file to load your key securely.

```python
import os
API_KEY = os.getenv("YOUTUBE_API_KEY")
```

## 📦 Installation

Install required packages using pip:

```bash
pip install youtube_transcript_api google-api-python-client scikit-learn matplotlib pandas
```

## 📝 How to Use

1. Run the notebook in Google Colab or locally.
2. Input a valid YouTube video link.
3. The notebook will:
   - Fetch video transcript and metadata
   - Perform topic modeling and detect chapter points
   - Output final chapters with timestamps and titles

## Project Structure

```
.
├── README.md               # Project documentation
├── transcript.csv          # My ouput transcript 
└── video_chaptering.py     # Coding file                  
```

## 📂 Output

- A CSV file containing transcript segments
- Console output showing identified chapters with timestamps and topic labels

## 📊 Visualizations

### 🔹 Text Length Distribution
![Text Length Distribution](https://github.com/user-attachments/assets/b63a1801-f1de-4197-bf16-fd6309f897ac)

### 🔹 Top 20 Most Common Words
![Most Common Words](https://github.com/user-attachments/assets/1a45c203-8025-4281-b99a-6e105f0b6764)


## 📸 Sample Output

```
Final Chapter Points with Names:
00:00:01 - Chapter 1: failure going way
00:01:02 - Chapter 2: bodybuilding program technique
00:02:03 - Chapter 3: better motion range
00:03:04 - Chapter 4: hypertrophy ll training
00:04:04 - Chapter 5: hard push sets
00:05:05 - Chapter 6: hypertrophy ll training
...
```
