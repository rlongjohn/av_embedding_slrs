import torch
import numpy as np
from more_itertools import windowed
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
    
class LuarModel:
    """Defines the LUAR embedding model.
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-CRUD")
        self.model = AutoModel.from_pretrained("rrivera1849/LUAR-CRUD", trust_remote_code=True)

    def calc_embedding_truncated(self, text):
        tokenized = self.tokenizer(
            [text],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        tokenized["input_ids"] = tokenized["input_ids"].reshape(1, 1, -1)
        tokenized["attention_mask"] = tokenized["attention_mask"].reshape(1, 1, -1)

        return self.model(**tokenized).detach().numpy()[0,]
    
    def calc_embedding_avg(self, text):

        tokenized = self.tokenizer(
            text, 
            max_length=512, 
            padding="do_not_pad", 
            truncation="do_not_truncate", 
            return_tensors="np")
        token_ids = tokenized['input_ids']
        token_mask = tokenized['attention_mask']
        # remove the start/end tokens (which are 0 and 2)
        text_tokens = token_ids[0][1:(len(token_ids[0])-1)]
        text_mask = token_mask[0][1:(len(token_mask[0])-1)]
        # get the chunks of length 510
        # the last chunk will have the padding token of 1 to fill it out at the end
        # the last mask will have the value of 0 corresponding to where there are padding tokens
        token_chunks = list(windowed(text_tokens, n=510, step=510-32, fillvalue=1))
        mask_chunks = list(windowed(text_mask, n=510, step=510-32, fillvalue=0))
        embeddings = []
        for i in range(len(token_chunks)):
            tokens = list(token_chunks[i])
            tokens.insert(0,0)
            tokens.append(2)
            mask = list(mask_chunks[i])
            mask.insert(0, 1)
            if (tokens[-1] == 1):
                mask.append(0)
            else:
                mask.append(1)

            tokens = torch.tensor(tokens).reshape(1, 1, -1)
            mask = torch.tensor(mask).reshape(1, 1, -1)

            tokenized_chunk = {'input_ids': tokens, 'attention_mask': mask}
            embeddings.append(self.model(**tokenized_chunk).detach().numpy()[0,])

        return np.mean(embeddings, axis = 0)
                

class CisrModel:
    """Defines the CISR embedding model.
    """
    def __init__(self):
        self.st_model = SentenceTransformer('AnnaWegmann/Style-Embedding')
        self.tokenizer = AutoTokenizer.from_pretrained('AnnaWegmann/Style-Embedding')
        self.model = AutoModel.from_pretrained('AnnaWegmann/Style-Embedding')

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def calc_embedding_truncated(self, text):
        return self.st_model.encode([text])[0]

    def calc_embedding_avg(self, text):

        tokenized = self.tokenizer(text, max_length=512, padding="do_not_pad", truncation="do_not_truncate", return_tensors="np")
        token_ids = tokenized['input_ids']
        token_mask = tokenized['attention_mask']
        # remove the start/end tokens (which are 0 and 2)
        text_tokens = token_ids[0][1:(len(token_ids[0])-1)]
        text_mask = token_mask[0][1:(len(token_mask[0])-1)]
        # get the chunks of length 510
        # the last chunk will have the padding token of 1 to fill it out at the end
        # the last mask will have the value of 0 corresponding to where there are padding tokens
        token_chunks = list(windowed(text_tokens, n=510, step=510-32, fillvalue=1))
        mask_chunks = list(windowed(text_mask, n=510, step=510-32, fillvalue=0))
        embeddings = []
        for i in range(len(token_chunks)):
            tokens = list(token_chunks[i])
            tokens.insert(0,0)
            tokens.append(2)
            mask = list(mask_chunks[i])
            mask.insert(0, 1)
            if (tokens[-1] == 1):
                mask.append(0)
            else:
                mask.append(1)

            tokens = torch.tensor(tokens).reshape(1, -1)
            mask = torch.tensor(mask).reshape(1, -1)

            tokenized_chunk = {'input_ids': tokens, 'attention_mask': mask}

            with torch.no_grad():
                model_output = self.model(**tokenized_chunk)
            
            embeddings.append(self.mean_pooling(model_output, tokenized_chunk['attention_mask']).detach().numpy()[0,])

        return np.mean(embeddings, axis = 0)