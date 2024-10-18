import pandas as pd
import numpy as np
import textstat
from spellchecker import SpellChecker
import language_tool_python
from sklearn.feature_extraction.text import CountVectorizer
# from tqdm.auto import tqdm
from tqdm.notebook import tqdm # if usign a notebook
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

class FeatureExtractor:
    """
    A class for extracting both statistical and content-based features from text data.
    """
    
    def __init__(self):
        """
        Initialize the FeatureExtractor class with required resources (Spacy, NLTK sentiment, SpellChecker, etc.).
        """
        self.nlp = spacy.load("en_core_web_md")  # For cohesion
        self.sia = SentimentIntensityAnalyzer()  # For sentiment analysis
        self.spell = SpellChecker()
        self.tool = language_tool_python.LanguageTool('en-US')

    def get_statistical_features(self, text):
        """
        Extract statistical features from the text, which are commonly used in Automated Essay Scoring (AES) research.

        Features extracted:
        (1) Total number of words.
        (2) Total number of characters.
        (3) Average number of words per sentence.
        (4) Total number of sentences.
        (5) Total number of paragraphs.
        (6) Total number of spelling mistakes.
        (7) Total number of grammar mistakes.
        (8) Average characters per word.
        (9) Average syllables per word.
        (10) Total number of punctuation marks.
        (11) Total number of unique words.
        (12) Type-Token Ratio (TTR): ratio of unique words over total words.
        (13) Total number of long words (words with more than 6 characters).
        (14) Total number of short sentences (less than 5 words).
        (15) Total number of complex sentences (with subordinating conjunctions).

        Additionally:
        (1) Flesch-Kincaid Grade Level: A readability score based on sentence length and word syllables.
        (2) Coleman-Liau Index: Another readability score based on sentence structure and word length.

        Parameters:
        text (str): The input text to extract features from.

        Returns:
        dict: A dictionary of extracted statistical features.
        """
        doc = self.nlp(text)
        words = text.split()
        
        features = {
            'word_count': len(words),
            'char_count': len(text),
            'avg_words_per_sentence': textstat.avg_sentence_length(text),
            'sentence_count': len(list(doc.sents)),
            'paragraph_count': text.count("\n") + 1,
            'spelling_mistakes': len(self.spell.unknown(words)),
            'grammar_mistakes': len(self.tool.check(text)),
            'avg_chars_per_word': textstat.avg_character_per_word(text),
            'avg_syllables_per_word': textstat.avg_syllables_per_word(text),
            'punctuation_count': sum([1 for char in text if char in '.,;!?']),
            'unique_word_count': len(set(words)),
            'ttr': len(set(words)) / len(words) if words else 0,
            'long_word_count': sum([1 for word in words if len(word) > 6]),
            'short_sentence_count': sum([1 for sent in doc.sents if len(sent.text.split()) < 5]),
            'complex_sentence_count': sum([1 for sent in doc.sents if any(token.dep_ == 'mark' for token in sent)]),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'coleman_liau_index': textstat.coleman_liau_index(text)
        }
        return features

    # def get_content_based_features(self, text):
    #     """
    #     Extract content-based features from the text.

    #     Features extracted:
    #     (1) Word Maturity: A placeholder for the average complexity of words based on length.
    #     (2) Sentiment: A sentiment score from NLTK's SentimentIntensityAnalyzer, ranging from -1 (negative) to 1 (positive).
    #     (3) Cohesion: The average semantic similarity between sentences using Spacy’s word vectors.

    #     Parameters:
    #     text (str): The input text to extract features from.

    #     Returns:
    #     dict: A dictionary of extracted content-based features.
    #     """
    #     if not isinstance(text, str) or pd.isna(text):
    #         return {
    #             'word_maturity': 0,  # Default value for word maturity
    #             'sentiment': 0,      # Default value for sentiment
    #             'cohesion': 0        # Default value for cohesion
    #         }
    #     doc = self.nlp(text)
        
        # # Word Maturity (dummy implementation)
        # word_maturity = np.mean([len(word) for word in text.split()])  # Placeholder for Word Maturity
        
        # # Sentiment analysis
        # sentiment = self.sia.polarity_scores(text)['compound']
        
        # try:
        #         # Cohesion (cosine similarity between sentences)
        #         vectors = [sent.vector for sent in doc.sents if sent.has_vector and len(sent.vector) > 0]
                
        #         # If vectors exist and are valid, compute cohesion
        #         if len(vectors) > 0:
        #             cohesion = np.mean([self.nlp.vocab.vectors.most_similar(vector)[0][1] for vector in vectors])
        #         else:
        #             cohesion = 0
        # except Exception as e:
        #     # Handle any exceptions and continue (e.g., missing vectors or dimensional issues)
        #     print(f"Error calculating cohesion: {e}")
        #     cohesion = 0  # Default to 0 if an error occurs

        # return {
        #     'word_maturity': word_maturity,
        #     'sentiment': sentiment,
        #     'cohesion': cohesion
        # }

    def extract_features(self, df, text_column):
        """
        Extract both statistical and content-based features from a DataFrame of text data.

        This function applies the get_statistical_features and get_content_based_features functions to each text entry in the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing a column of text data.
        text_column (str): The name of the column containing the text data.

        Returns:
        pd.DataFrame: A new DataFrame with both statistical and content-based features.
        """
        df = df.copy()
        df = df.reset_index(drop=True)
        tqdm.pandas()

        # Extract statistical features
        stats = df[text_column].progress_apply(self.get_statistical_features)
        # stats_df = stats_df.add_suffix('_stats')
        # df = pd.concat([df, stats_df], axis=1)
        stats_df = pd.DataFrame(stats.tolist())
        # # Extract content-based features
        # content = df[text_column].progress_apply(self.get_content_based_features)
        # content_df = pd.DataFrame(content.tolist())
        # df = pd.concat([df, content_df], axis=1)

        return stats_df
