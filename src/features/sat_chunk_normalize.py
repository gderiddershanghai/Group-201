import pandas as pd
import re

def rescale_scores_with_categories(df, score_column, output_range=(1, 10)):
    """
    Rescales scores and categorizes them into 'low', 'medium', and 'high' based on the rescaled values.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the scores to rescale.
        score_column (str): The column name containing the scores to be rescaled.
        output_range (tuple): The desired output range (min, max) for rescaling.

    Returns:
        pd.DataFrame: A DataFrame containing essay_id, essay_set, rescaled scores, and categories.
    """
    # Unpack the desired range
    range_min, range_max = output_range
    
    # Calculate min and max for each essay_set
    score_ranges = df.groupby('essay_set')[score_column].agg(['min', 'max']).reset_index()
    
    # Merge the min-max values back into the original DataFrame
    df = df.merge(score_ranges, on='essay_set', how='left')
    
    # Perform Min-Max scaling to the specified range
    df['rescaled_score'] = range_min + (range_max - range_min) * (
        (df[score_column] - df['min']) / (df['max'] - df['min'])
    )
    
    # Add categories based on rescaled_score
    df['low_med_hi'] = pd.cut(
        df['rescaled_score'],
        bins=[range_min, 5, 7.5, range_max],
        labels=['low', 'medium', 'high'],
        include_lowest=True
    )
    
    # Add numeric categories
    df['low_med_hi_numeric'] = df['low_med_hi'].map({'low': 1, 'medium': 2, 'high': 3})
    
    # Select and return the desired columns
    return df[['essay_id', 'essay_set', 'essay' , 'rescaled_score', 'low_med_hi', 'low_med_hi_numeric']]


# import pandas as pd
# import re

# class Chunker:
#     def __init__(self, data):
#         """
#         Initializes the Chunker class with the provided data.

#         Parameters:
#         data (pd.DataFrame): The DataFrame containing the dataset with essays and metadata.
#         """
#         # Filter out essays with fewer than 25 words
#         self.data = data[data['essay'].str.split().apply(len) >= 25].reset_index(drop=True)
#         self.chunked_df = None

#         # Lists to store processed data
#         self.essay_ids = []  # Essay IDs
#         self.essay_sets = []  # Essay sets
#         self.paragraphs = []  # Processed paragraphs
#         self.rescaled_scores = []  # Rescaled scores
#         self.low_med_hi = []  # Low/medium/high categories
#         self.low_med_hi_numeric = []  # Numeric categories

#     def split_long_paragraph(self, paragraph, max_words=275):
#         """
#         Splits a long paragraph into two parts if it exceeds the maximum word count.

#         Parameters:
#         paragraph (str): The paragraph to split.
#         max_words (int): The maximum word count before splitting.

#         Returns:
#         list: A list of one or two paragraphs.
#         """
#         words = paragraph.split()
#         if len(words) <= max_words:
#             return [paragraph]  # No split needed

#         # Split into sentences and divide into two halves
#         sentences = re.split(r'(?<=[.!?])\s+', paragraph)
#         halfway_point = len(sentences) // 2

#         first_half = " ".join(sentences[:halfway_point]).strip()
#         second_half = " ".join(sentences[halfway_point:]).strip()

#         return [first_half, second_half]

#     def chunk_text_by_paragraph(self, text, min_length=100):
#         """
#         Splits the provided essay text into chunks based on sentence boundaries
#         and aggregates sentences until the combined word count exceeds a minimum length.

#         Parameters:
#         text (str): The input essay text to chunk.
#         min_length (int): The minimum acceptable length for a chunk (in words).

#         Returns:
#         list: A list of processed text chunks.
#         """
#         # Split text into sentences
#         sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#         chunks = []
#         current_chunk = []
#         current_word_count = 0

#         for sentence in sentences:
#             # Count words in the current sentence
#             sentence_word_count = len(sentence.split())
#             current_chunk.append(sentence.strip())
#             current_word_count += sentence_word_count

#             # If the chunk's total word count exceeds the minimum length
#             if current_word_count >= min_length:
#                 # Combine sentences into a single chunk
#                 combined_chunk = " ".join(current_chunk)

#                 # Check if the chunk exceeds the max length (350 words)
#                 for sub_chunk in self.split_long_paragraph(combined_chunk):
#                     chunks.append(sub_chunk)

#                 # Reset for the next chunk
#                 current_chunk = []
#                 current_word_count = 0

#         # Handle the remaining chunk if it meets the minimum length
#         if current_word_count >= min_length:
#             combined_chunk = " ".join(current_chunk)
#             for sub_chunk in self.split_long_paragraph(combined_chunk):
#                 chunks.append(sub_chunk)

#         return chunks

#     def get_chunks(self, min_length=100):
#         """
#         Processes the dataset and breaks each essay into chunks, collecting metadata.

#         Parameters:
#         min_length (int): Minimum acceptable length for a chunk (in words).

#         Returns:
#         pd.DataFrame: A DataFrame with chunked data and metadata.
#         """
#         # Iterate through dataset rows
#         for row in self.data.itertuples():
#             essay_id = getattr(row, 'essay_id')
#             essay_set = getattr(row, 'essay_set')
#             essay = getattr(row, 'essay')
#             rescaled_score = getattr(row, 'rescaled_score')
#             low_med_hi = getattr(row, 'low_med_hi')
#             low_med_hi_numeric = getattr(row, 'low_med_hi_numeric')

#             # Chunk the essay text
#             paragraphs = self.chunk_text_by_paragraph(essay, min_length)

#             # Append metadata for each chunk
#             for para in paragraphs:
#                 self.essay_ids.append(essay_id)
#                 self.essay_sets.append(essay_set)
#                 self.paragraphs.append(para)
#                 self.rescaled_scores.append(rescaled_score)
#                 self.low_med_hi.append(low_med_hi)
#                 self.low_med_hi_numeric.append(low_med_hi_numeric)

#         # Create a DataFrame from the chunked data
#         self.chunked_df = pd.DataFrame({
#             'essay_id': self.essay_ids,
#             'essay_set': self.essay_sets,
#             'paragraph': self.paragraphs,
#             'rescaled_score': self.rescaled_scores,
#             'low_med_hi': self.low_med_hi,
#             'low_med_hi_numeric': self.low_med_hi_numeric
#         })

#         return self.chunked_df

import pandas as pd
import re

class Chunker:
    def __init__(self, data):
        """
        Initializes the Chunker class with the provided data.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the dataset with essays and metadata.
        """
        # Filter out essays with fewer than 50 words
        self.data = data[data['essay'].str.split().apply(len) >= 50].reset_index(drop=True)
        self.chunked_df = None

        # Lists to store processed data
        self.essay_ids = []  # Essay IDs
        self.essay_sets = []  # Essay sets
        self.paragraphs = []  # Processed paragraphs
        self.rescaled_scores = []  # Rescaled scores
        self.low_med_hi = []  # Low/medium/high categories
        self.low_med_hi_numeric = []  # Numeric categories
        self.chunk = []  # Original chunk column

    def split_into_sentences(self, text):
        """
        Splits text into sentences based on punctuation.
        
        Parameters:
        text (str): The input text to split.

        Returns:
        list: A list of sentences.
        """
        return re.split(r'(?<=[.!?])\s+', text.strip())

    def chunk_text_by_fraction(self, text, threshold=300, num_chunks=3, min_chunk_length=75):
        """
        Splits the essay text into chunks respecting sentence boundaries.

        Parameters:
        text (str): The input essay text to chunk.
        threshold (int): The word count threshold to decide the number of chunks.
                         Essays shorter than this will be split into two chunks.
        num_chunks (int): Default number of chunks for longer essays.
        min_chunk_length (int): Minimum length of a chunk in words. Chunks shorter than this are merged.

        Returns:
        list: A list of processed text chunks.
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected text to be a string, but got {type(text)}")

        sentences = self.split_into_sentences(text)
        total_word_count = sum(len(sentence.split()) for sentence in sentences)
        actual_num_chunks = 2 if total_word_count < threshold else num_chunks

        # Calculate target chunk size
        chunk_size = max(1, total_word_count // actual_num_chunks)
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            if current_word_count + sentence_word_count >= chunk_size and current_word_count >= min_chunk_length:
                # Finalize current chunk if it has enough words
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            # Add sentence to the current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

        # Add remaining sentences as the last chunk
        if current_chunk:
            if chunks and current_word_count < min_chunk_length:
                # Merge with the previous chunk if too short
                chunks[-1] += " " + " ".join(current_chunk)
            else:
                chunks.append(" ".join(current_chunk))

        return chunks

    def get_chunks(self, num_chunks=3, threshold=300, min_chunk_length=75):
        """
        Processes the dataset and breaks each essay into chunks, collecting metadata.

        Parameters:
        num_chunks (int): Number of chunks to split longer essays into.
        threshold (int): The word count threshold for splitting shorter essays into two chunks.
        min_chunk_length (int): Minimum acceptable chunk length in words.

        Returns:
        pd.DataFrame: A DataFrame with chunked data and metadata, including the original `chunk` column.
        """
        if not isinstance(num_chunks, int) or num_chunks <= 0:
            raise ValueError(f"num_chunks must be a positive integer, got {num_chunks}")

        # Reset lists to avoid duplicates
        self.essay_ids = []
        self.essay_sets = []
        self.paragraphs = []
        self.rescaled_scores = []
        self.low_med_hi = []
        self.low_med_hi_numeric = []
        self.chunk = []

        # Iterate through dataset rows
        for row in self.data.itertuples():
            essay_id = getattr(row, 'essay_id')
            essay_set = getattr(row, 'essay_set')
            essay = getattr(row, 'essay')
            rescaled_score = getattr(row, 'rescaled_score')
            low_med_hi = getattr(row, 'low_med_hi')
            low_med_hi_numeric = getattr(row, 'low_med_hi_numeric')
            original_chunk = getattr(row, 'chunk')  # Preserve the original chunk value

            # Ensure essay is a string
            essay = str(essay)

            # Chunk the essay text
            paragraphs = self.chunk_text_by_fraction(essay, threshold=threshold, num_chunks=num_chunks, min_chunk_length=min_chunk_length)

            # Assign chunks
            for para in paragraphs:
                self.essay_ids.append(essay_id)
                self.essay_sets.append(essay_set)
                self.paragraphs.append(para)
                self.rescaled_scores.append(rescaled_score)
                self.low_med_hi.append(low_med_hi)
                self.low_med_hi_numeric.append(low_med_hi_numeric)
                self.chunk.append(original_chunk)

        # Create a DataFrame from the chunked data
        self.chunked_df = pd.DataFrame({
            'essay_id': self.essay_ids,
            'essay_set': self.essay_sets,
            'chunk': self.chunk,  # Original chunk column
            'paragraph': self.paragraphs,
            'rescaled_score': self.rescaled_scores,
            'low_med_hi': self.low_med_hi,
            'low_med_hi_numeric': self.low_med_hi_numeric
        })

        return self.chunked_df
