import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import torch


class BertTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bert_tokenizer, bert_model, *, max_length=140, embedding_func=None, use_attention_mask=False):
        self.tokenizer = bert_tokenizer
        self.model = bert_model
        self.model.eval()
        self.max_length = max_length
        self.embedding_func = embedding_func
        self.use_attention_mask = use_attention_mask

        if self.embedding_func is None:
            self.embedding_func = lambda x: x[0][:, 0, :].squeeze()

    def _tokenize(self, text):
        tokenization_results = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=self.max_length
        )
        tokenized_data = tokenization_results['input_ids']
        if self.use_attention_mask:
            attention_mask = tokenization_results['attention_mask']
        else:
            attention_mask = [1] * len(tokenized_data)
        return (
            torch.tensor(tokenized_data).unsqueeze(0),
            torch.tensor(attention_mask).unsqueeze(0),
        )

    def _tokenize_and_predict(self, text):
        tokenized, attention_mask = self._tokenize(text)
        embeddings = self.model(tokenized, attention_mask)
        return self.embedding_func(embeddings)

    def transform(self, text_entries):
        if isinstance(text_entries, pd.Series):
            text_entries = text_entries.tolist()

        with torch.no_grad():
            return torch.stack([self._tokenize_and_predict(text) for text in text_entries])

    def fit(self, entries, labels=None):
        return self
    