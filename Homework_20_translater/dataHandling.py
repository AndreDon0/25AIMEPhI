import numpy as np
import torch
from pandas import Series
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from typing import List, Union
from torch.nn.utils.rnn import pad_sequence


class CaesarCipherHandler:
    def __init__(self, alphabet: List[str], shift: int):
        self.alphabet = alphabet
        self.shift = shift
    
    def cipher(self, text: str) -> str:
        return "".join([self.alphabet[(self.alphabet.index(char) + self.shift) % len(self.alphabet)] for char in text])

    def decipher(self, text: str) -> str:
        return "".join([self.alphabet[(self.alphabet.index(char) - self.shift) % len(self.alphabet)] for char in text])

class MyTokenizer:
    def __init__(self, alphabet: List[str]):
        self.alphabet = alphabet
        self.token_to_index = {token: index for index, token in enumerate(alphabet, start=0)}
        self.index_to_token = {index: token for index, token in enumerate(alphabet, start=0)}

    def tokenize(self, text: str) -> List[int]:
        return [self.token_to_index[token] for token in text]

    def detokenize(self, tokens: Union[List[int], torch.Tensor]) -> str:
        if torch.is_tensor(tokens):
            tokens = tokens.detach().cpu().flatten().tolist()
        return "".join(self.index_to_token[int(t)] for t in tokens)
    
    def __len__(self):
        return len(self.alphabet)

class TrainDataset(Dataset):
    def __init__(
        self,
        data: Series,
        handler: CaesarCipherHandler,
        tokenizer: MyTokenizer,
    ):
        self.handler = handler
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]
        cipher_text = self.handler.cipher(text)
        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_cipher_text = self.tokenizer.tokenize(cipher_text)
        n = len(self.tokenizer)
        one_hot_text = np.eye(n, dtype=np.float32)[tokenized_text]

        target_ids = torch.tensor(tokenized_cipher_text, dtype=torch.long)
        return torch.from_numpy(one_hot_text), target_ids

class TestDataset(Dataset):
    def __init__(
        self,
        data: Series,
        handler: CaesarCipherHandler,
        tokenizer: MyTokenizer,
    ):
        self.handler = handler
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]
        tokenized_text = self.tokenizer.tokenize(text)
        n = len(self.tokenizer)
        one_hot_text = np.eye(n, dtype=np.float32)[tokenized_text]
        return torch.from_numpy(one_hot_text)

def get_loaders(
    data: Series,
    handler: CaesarCipherHandler,
    tokenizer: MyTokenizer,
    test_size: float = 0.2,
    batch_size: int = 32,
):
    train_dataset, test_dataset = train_test_split(data, test_size=test_size, random_state=42)
    train_dataset = TrainDataset(train_dataset, handler, tokenizer)
    test_dataset = TestDataset(test_dataset, handler, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_train_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_test_batch,
    )
    return train_loader, test_loader


def collate_train_batch(batch):
    texts = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0.0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-100)
    return padded_texts, padded_targets


def collate_test_batch(batch):
    padded_texts = pad_sequence(batch, batch_first=True, padding_value=0.0)
    return padded_texts