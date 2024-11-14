import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

class Chunker():
    def __init__(self, data, data_set_name='IELTS'):
        self.ratings = []
        self.questions = []
        self.paragraphs = []
        self.original_index = []
        self.data_source = []
        
        self.data = data
        self.data_set_name = data_set_name
        self.chunked_df = None

    def chunk_text_by_paragraph(self, text, min_length=10, max_length=125):
        """
        Splits the provided essay text into paragraphs, ensuring each paragraph
        is within the specified length. Short paragraphs are merged with adjacent paragraphs,
        and long paragraphs are split into smaller chunks based on sentence boundaries.

        Parameters:
        text (str): The input essay text to chunk. The text is split by newlines into paragraphs.
        min_length (int): The minimum acceptable length for a paragraph (in words).
        max_length (int): The maximum acceptable length for a paragraph (in words).

        Returns:
        list: A list of paragraphs, where each element is a chunked paragraph from the essay.
        """
        # Step 1: Split by newlines to get initial paragraphs
        paragraphs = [para.strip() for para in text.split('\n') if para.strip()]
        refined_paragraphs = []

        for paragraph in paragraphs:
            words = paragraph.split()
            
            # Step 2: Handle short paragraphs by merging
            if len(words) < min_length:
                if refined_paragraphs:
                    refined_paragraphs[-1] += " " + paragraph
                else:
                    refined_paragraphs.append(paragraph)
            elif len(words) > max_length:
                # Step 3: Handle long paragraphs by splitting into chunks
                sentences = sent_tokenize(paragraph)
                current_chunk = []
                current_chunk_length = 0

                for sentence in sentences:
                    sentence_length = len(sentence.split())

                    # Add sentence to the current chunk if it fits
                    if current_chunk_length + sentence_length <= max_length:
                        current_chunk.append(sentence)
                        current_chunk_length += sentence_length
                    else:
                        # If the current chunk is within limits, save it
                        if min_length <= current_chunk_length <= max_length:
                            refined_paragraphs.append(" ".join(current_chunk))
                            current_chunk = [sentence]
                            current_chunk_length = sentence_length
                        else:
                            # If the chunk remains too long, split by words to enforce the max limit
                            words_in_chunk = " ".join(current_chunk).split()
                            for i in range(0, len(words_in_chunk), max_length):
                                refined_paragraphs.append(" ".join(words_in_chunk[i:i + max_length]))
                            current_chunk = [sentence]
                            current_chunk_length = sentence_length

                # Append the final chunk if it's within limits
                if current_chunk:
                    words_in_chunk = " ".join(current_chunk).split()
                    for i in range(0, len(words_in_chunk), max_length):
                        refined_paragraphs.append(" ".join(words_in_chunk[i:i + max_length]))
            else:
                # If the paragraph is within the acceptable range, add it directly
                refined_paragraphs.append(paragraph)

        # Final pass to merge any remaining short paragraphs
        final_paragraphs = []
        i = 0
        while i < len(refined_paragraphs):
            paragraph = refined_paragraphs[i]
            words = paragraph.split()
            if len(words) < min_length and i + 1 < len(refined_paragraphs):
                # Merge with the next paragraph if the current one is too short
                paragraph = paragraph + " " + refined_paragraphs[i + 1]
                i += 1  # Skip the next paragraph as it's merged with the current one
            final_paragraphs.append(paragraph)
            i += 1

        return final_paragraphs

    def get_chunks(self):
        """
        Processes the entire dataset and breaks down each essay into smaller chunks (paragraphs).
        The method collects relevant metadata such as ratings, questions, and original indices
        for each paragraph and stores them in lists. A DataFrame is generated with the processed
        data, where each row corresponds to a paragraph from the dataset.

        Returns:
        pd.DataFrame: A DataFrame containing the chunked data.
        """
        if self.data_set_name == 'IELTS':
            question_col = 'Question'
            essay_col = 'Essay'
            score_col = 'Overall'
            data_type = 'Task_Type'
            
        word_counts = []
        for row in self.data.itertuples():
            rating = row._asdict()[score_col]
            original_index = row.Index
            question = row._asdict()[question_col]
            essay = row._asdict()[essay_col]
            task_type = row._asdict()[data_type]

            # Split the essay into individual paragraphs using the helper method
            paragraphs = self.chunk_text_by_paragraph(essay)

            for para in paragraphs:
                self.ratings.append(rating)
                self.questions.append(question)
                self.paragraphs.append(para)
                self.original_index.append(original_index)
                self.data_source.append(task_type)
                word_counts.append(len(para.split()))

        # Create a DataFrame from the collected chunked data
        self.chunked_df = pd.DataFrame({
            'Original_Index': self.original_index,
            'Data_Source': self.data_source,
            'Question': self.questions,
            'Paragraph': self.paragraphs,
            'Rating': self.ratings,
            'Number_Of_Words': word_counts
        })
        
        return self.chunked_df
