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
            mask = list(mask_chunks[i])

            # add start token to each individual chunk
            tokens.insert(0,0)
            mask.insert(0, 1)

            # last chunk might not be full
            if i == len(token_chunks) - 1:
                # remove None filler elements from the windowed call
                tokens = [token for token in tokens if token is not None]
                mask = [m for m in mask if m is not None]
                # add end token to each individual chunk
                tokens.append(2)
                mask.append(1)
                # if chunk isn't full add the padding tokens and corresponding mask 0
                if len(tokens) < 512:
                    tokens += [1] * (512 - len(tokens))
                    mask += [0] * (512 - len(mask))
            else:
                # add end token to each individual chunk
                tokens.append(2)
                mask.append(1)

            tokens = torch.tensor(tokens).reshape(1, 1, -1)
            mask = torch.tensor(mask).reshape(1, 1, -1)

            tokenized_chunk = {'input_ids': tokens, 'attention_mask': mask}
            embeddings.append(self.model(**tokenized_chunk).detach().numpy()[0,])

        return np.mean(embeddings, axis = 0)
    
    def calc_embedding_window(self, text):
        tokenized_text = self.tokenizer(text,
                           max_length=32, 
                           padding="do_not_pad", 
                           truncation="do_not_truncate", 
                           return_tensors="np")
        
        # retrieve tokenized info
        text_tokens = tokenized_text['input_ids'][0]
        text_mask = tokenized_text['attention_mask'][0]
       
        # remove the start and end tokens; 0 and 2, respectively (will add back later)
        text_tokens = text_tokens[1:(len(text_tokens)-1)]
        text_mask = text_mask[1:(len(text_mask)-1)]
        
        # grab 30-token chunks of text tokens and mask info (30 bc we'll add the start/end tokens later to get 32)
        token_chunks = list(windowed(text_tokens, n=30, step=30))
        mask_chunks = list(windowed(text_mask, n=30, step=30))
        for i in range(len(token_chunks)):
            tokens = list(token_chunks[i])
            mask = list(mask_chunks[i])

            # add start token to each individual chunk
            tokens.insert(0,0)
            mask.insert(0, 1)

            # last chunk might not be full
            if i == len(token_chunks) - 1:
                # remove None filler elements from the windowed call
                tokens = [token for token in tokens if token is not None]
                mask = [m for m in mask if m is not None]
                # add end token to each individual chunk
                tokens.append(2)
                mask.append(1)
                # if chunk isn't full add the padding tokens and corresponding mask 0
                if len(tokens) < 32:
                    tokens += [1] * (32 - len(tokens))
                    mask += [0] * (32 - len(mask))
            else:
                # add end token to each individual chunk
                tokens.append(2)
                mask.append(1)
            
            # store modified chunk elements
            token_chunks[i] = tokens
            mask_chunks[i] = mask

        # convert to numpy array from list
        token_chunks = np.array(token_chunks)
        mask_chunks = np.array(mask_chunks)
        # total number of 32-token chunks
        n_chunks = token_chunks.shape[0]

        # reshape the chunks
        full_token_chunks = torch.tensor(token_chunks).reshape(1, n_chunks, -1)
        full_mask_chunks = torch.tensor(mask_chunks).reshape(1, n_chunks, -1)

        # format new tokenized text
        tokenized_windows = {'input_ids': full_token_chunks, 'attention_mask': full_mask_chunks}

        # final embedding
        return self.model(**tokenized_windows).detach().numpy()[0,]
                

class CisrModel:
    """Defines the CISR embedding model.

    Implementation follows from https://huggingface.co/AnnaWegmann/Style-Embedding
    including the mean_pooling function

    calc_embedding_avg does a sliding window over tokens with 32 tokens of overlap
    Windows are size 512 (510 tokens plus start and end token)

    The truncated model uses the "encode" class method

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
        token_chunks = list(windowed(text_tokens, n=510, step=510-32))
        mask_chunks = list(windowed(text_mask, n=510, step=510-32))
        embeddings = []
        for i in range(len(token_chunks)):
            tokens = list(token_chunks[i])
            tokens = [t for t in tokens if t is not None]
            tokens.insert(0,0)
            tokens.append(2)
            mask = list(mask_chunks[i])
            mask = [m for m in mask if m is not None]
            mask.insert(0, 1)
            mask.append(1)

            tokens = torch.tensor(tokens).reshape(1, -1)
            mask = torch.tensor(mask).reshape(1, -1)

            tokenized_chunk = {'input_ids': tokens, 'attention_mask': mask}

            with torch.no_grad():
                model_output = self.model(**tokenized_chunk)
            
            embeddings.append(self.mean_pooling(model_output, tokenized_chunk['attention_mask']).detach().numpy()[0,])

        return np.mean(embeddings, axis = 0)