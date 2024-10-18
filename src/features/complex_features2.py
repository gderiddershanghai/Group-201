from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, LongformerTokenizer, LongformerModel
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

class ComplexFeatureExtractor:
    """
    A transformer-style class that processes text and extracts embeddings from pre-trained models (BERT, RoBERTa, Longformer).
    Follows the stateless design, similar to scikit-learn's transform/predict methods.
    """
    
    # Class-level variables (shared across all instances)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models and tokenizers once as class variables
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
    
    longformer_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    longformer_model = LongformerModel.from_pretrained('allenai/longformer-base-4096').to(device)

    def __init__(self):
        """
        Initialize the ComplexFeatureExtractor class. No internal state is maintained.
        """
        pass

    @staticmethod
    def chunk_text_by_paragraph(text):
        """
        Splits the text into paragraphs for better context extraction.
        Each paragraph will be treated as an individual data point.

        Parameters:
        text (str): The input text to chunk.

        Returns:
        list: A list of paragraphs.
        """
        paragraphs = [para.strip() for para in text.split('\n') if para.strip()]
        return paragraphs

    @classmethod
    def get_bert_embedding(cls, text):
        """
        Extracts embeddings from BERT for a given text (paragraph).

        Parameters:
        text (str): Input text (paragraph).

        Returns:
        np.ndarray: Mean-pooled embedding for the text.
        """
        encoded_input = cls.bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(cls.device)
        with torch.no_grad():
            output = cls.bert_model(**encoded_input)
        mean_embedding = output.last_hidden_state.mean(1).cpu().numpy()
        return mean_embedding

    @classmethod
    def get_roberta_embedding(cls, text):
        """
        Extracts embeddings from RoBERTa for a given text (paragraph).

        Parameters:
        text (str): Input text (paragraph).

        Returns:
        np.ndarray: Mean-pooled embedding for the text.
        """
        encoded_input = cls.roberta_tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(cls.device)
        with torch.no_grad():
            output = cls.roberta_model(**encoded_input)
        mean_embedding = output.last_hidden_state.mean(1).cpu().numpy()
        return mean_embedding

    @classmethod
    def get_longformer_embedding(cls, text):
        """
        Extracts embeddings from Longformer for a given text (paragraph).

        Parameters:
        text (str): Input text (paragraph).

        Returns:
        np.ndarray: Mean-pooled embedding for the text.
        """
        encoded_input = cls.longformer_tokenizer(text, return_tensors='pt', truncation=True, max_length=1200, padding=True).to(cls.device)
        with torch.no_grad():
            output = cls.longformer_model(**encoded_input)
        mean_embedding = output.last_hidden_state.mean(1).cpu().numpy()
        return mean_embedding

    def transform(self, df, text_column, model='bert'):
        """
        Extracts embeddings from the specified model (BERT, RoBERTa, or Longformer) for each paragraph in the dataset.
        Returns the embeddings for each paragraph in the form of a list of numpy arrays.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the text data.
        text_column (str): The column containing the text.
        model (str): The model to use for embeddings ('bert', 'roberta', 'longformer').

        Returns:
        list: A list of embeddings (numpy arrays) for each paragraph in the dataset.
        """
        embeddings_with_index = []
        model_function = {
            'bert': self.get_bert_embedding,
            'roberta': self.get_roberta_embedding,
            'longformer': self.get_longformer_embedding
        }.get(model)

        if model_function is None:
            raise ValueError(f"Unsupported model '{model}'. Choose from 'bert', 'roberta', or 'longformer'.")

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = row[text_column]
            paragraphs = self.chunk_text_by_paragraph(text)

            # Extract embeddings for each paragraph
            for para in paragraphs:
                embedding = model_function(para)
                embeddings_with_index.append((idx, embedding))  # Store essay index and the paragraph embedding

        return embeddings_with_index

