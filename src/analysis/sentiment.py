from transformers import pipeline
import torch # Transformers often need torch

class SentimentAnalyzer:
    def __init__(self, config):
        self.model_name = config['sentiment']['model_name']
        print(f"Initializing Sentiment Analyzer (Model: {self.model_name})...")
        try:
            # Use device=-1 for CPU explicitly
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=-1 # Use CPU
            )
            print("Sentiment Analyzer initialized.")
        except Exception as e:
             print(f"Error initializing sentiment pipeline: {e}")
             print("Ensure 'transformers' and 'torch' are installed.")
             # Decide if sentiment is critical - maybe set pipeline to None?
             self.sentiment_pipeline = None
             print("Sentiment analysis disabled due to initialization error.")

    def analyze(self, text):
        """Analyzes sentiment of the text, returns 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'."""
        if not self.sentiment_pipeline or not text:
            return "NEUTRAL" # Default if disabled or no text

        try:
            results = self.sentiment_pipeline(text)
            # [{'label': 'POSITIVE', 'score': 0.999}]
            # Return the label directly
            return results[0]['label']
        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            return "NEUTRAL" # Default on error
