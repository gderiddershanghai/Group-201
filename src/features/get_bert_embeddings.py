from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm.auto import tqdm

class BertFeatureExtractor:
    """
    A class to extract BERT embeddings (mean, [CLS] token, or pooler output) from pre-chunked text data.
    """

    def __init__(self):
        """
        Initializes the BertFeatureExtractor with pre-loaded BERT model and tokenizer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def get_bert_embedding(self, text, embedding_type='mean'):
        """
        Extracts BERT embeddings for a given text (paragraph) with the specified embedding type.

        Parameters:
        text (str): Input text (paragraph).
        embedding_type (str): The type of embedding to extract ('mean', '[CLS]', or 'pooler_output').

        Returns:
        np.ndarray: The selected embedding for the text.
        """
        encoded_input = self.bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            output = self.bert_model(**encoded_input)
        
        if embedding_type == 'mean':
            # Mean-pooling across all tokens
            embedding = output.last_hidden_state.mean(1).cpu().numpy()
        elif embedding_type == '[CLS]':
            # [CLS] token embedding (first token)
            embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
        elif embedding_type == 'pooler_output':
            # Pooler output (used for classification tasks)
            embedding = output.pooler_output.cpu().numpy()
        else:
            raise ValueError(f"Unsupported embedding type '{embedding_type}'. Choose from 'mean', '[CLS]', or 'pooler_output'.")
        
        return embedding

    def transform(self, df, embedding_type='pooler_output'):
            """
            Extracts BERT embeddings of the specified type for each paragraph in the preprocessed DataFrame.
            Returns a list of numpy arrays of embeddings for each paragraph.

            Parameters:
            ----------
            df : pd.DataFrame
                The DataFrame containing the chunked text data in the 'Essay' column.
            embedding_type : str, optional
                The type of embedding to extract. Options are 'mean', '[CLS]', and 'pooler_output'.
                Default is 'mean'.

            Returns:
            -------
            list of np.ndarray
                A list of numpy arrays containing the embeddings for each paragraph in the dataset.
            """
            embeddings = []

            for _, row in tqdm(df.iterrows(), total=len(df)):
                text = row['Paragraph']  # Paragraph is stored in the 'Essay' column

                # Extract embedding for the paragraph
                embedding = self.get_bert_embedding(text, embedding_type)
                embeddings.append(embedding)  # Store only the paragraph embedding

            return embeddings
