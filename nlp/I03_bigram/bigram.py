import tiktoken
import json
import torch

from typing import List
from typing import Set
from util.file_utils import get_file_names

# Configuration
corpus_folder  = "corpus"
end_token      = "<|endoftext|>"
tokenizer_name = 'cl100k_base'

# Initialization
tokenizer = tiktoken.get_encoding(tokenizer_name)
bigrams_dict = {} # dictionary of bigram
vocabulary: Set[str] = None


def encode(text) -> List[int]:
    return tokenizer.encode(text, allowed_special={end_token})


def decode(tokens: List[int]) -> List[str]:
    return tokenizer.decode(tokens)


def decode_single_token(tokens: List[int]) -> List[str]:
    return [decode([tk]) for tk in tokens]


def compute_bigram_frequency(encoded_txt):
    for tk1, tk2 in zip(encoded_txt, encoded_txt[1:]):
        b = (tk1, tk2)
        bigrams_dict[b] = bigrams_dict.get(b, 0) + 1  
    return bigrams_dict
    

def decode_bigram_freq(bigrams_dict):
    if len(bigrams_dict) > 0:
        decode_bigrams_dict = {}
        for key, value in bigrams_dict.items():
            decoded_key = (decode_bigram(key))
            decode_bigrams_dict[decoded_key] = value
        return decode_bigrams_dict
    else:
        return None

        
def decode_bigram(bigram_tokens: Set[int]) -> Set[str]:
    if bigram_tokens:
        bigrams = list(bigram_tokens)
        b1 = decode([bigrams[0]])
        b2 = decode([bigrams[1]])
        return (b1, b2)
    else:
        return None

        
def decode_bigrams(bigram_tokens: List[Set[int]]) -> List[Set[str]]:
    if len(bigram_tokens) > 0:
        decode_bigrams_list = []
        for b in bigram_tokens:
            decode_bigrams_list.append(decode_bigram(b))
        return decode_bigrams_list
    else:
        return None

        
if __name__ == '__main__':
    file_names = get_file_names(corpus_folder)
    texts = []
    for filename in file_names[:1]:
        with open(f"{corpus_folder}/{filename}", "r", encoding='utf-8') as file:
            print(f"loading... ({filename})")
            data = json.load(file);
            text = data.get("text", "")
            texts.append(text + end_token)  # Append text and add space


    # Total of tokens
    cod_tokens = encode(texts[0])
    print("Total of tokens:", len(cod_tokens))  
    # Show text encode and decode
    print(texts[0][:80])
    print(cod_tokens[:20], '...')
    print(decode_single_token(cod_tokens)[:10], '...')
    print(decode(cod_tokens[:20]), '...', '\n')

    # Create a set of bigrams_dict and its frequencies
    texts_tokens = []
    for txt in texts:
        cod_tokens = encode(txt)
        txt_tokens = decode_single_token(cod_tokens)
        if vocabulary:
            vocabulary = vocabulary.union(txt_tokens)
        else:
            vocabulary = set(txt_tokens)
        bigrams_dict = compute_bigram_frequency(cod_tokens)   
        texts_tokens.append(txt_tokens)

    # Show bigram
    print("Vocalubary size:", len(vocabulary))
    print('Bigrams:')
    print(list(bigrams_dict.keys())[:5], '...')  
    decoded_bigrams_list = decode_bigrams(list(bigrams_dict.keys()))
    print(list(decoded_bigrams_list)[:5], '...')

    # Show part of the bigrams       
    print('Bigrams frenquecies:')  
    bigram_list = list(bigrams_dict.items())
    print(bigram_list[:5], '...')   
    tkn_freq = decode_bigram_freq(bigrams_dict)
    tkn_freq = list(tkn_freq.items())
    print(tkn_freq[:5], '...')   

    # Sorted bigrams by frequency
    print('Bigrams frenquecies:')  
    bigram_list = sorted(bigrams_dict.items(), key = lambda value: value[1], reverse=True)
    print(bigram_list[:5], '...')   
    tkn_freq = decode_bigram_freq(bigrams_dict)
    tkn_freq = sorted(tkn_freq.items(), key = lambda value: value[1], reverse=True)
    print(tkn_freq[:5], '...', '\n')   

    # Maps it token (string) to a integer (sequencialy)
    sort_voc = sorted(vocabulary)
    stoi = {s:i for i, s in enumerate(sort_voc)}  # stoi - string (word) to integer    
    itos = {i:s for i, s in enumerate(stoi)}
    print("Dicionaries: 'stoi' and 'itos'")
    print(list(stoi.items())[:10], '...')
    print(list(itos.items())[:10], '...', '\n')
    
    # Create table of frequencies for bigrams
    print("Frequency table:")
    total_tokens = len(stoi)
    N = torch.zeros((total_tokens, total_tokens), dtype=torch.int32)
    for text_tkn in texts_tokens:
        for tk1, tk2 in zip(text_tkn, text_tkn[1:]):      
          #print(f"{tk1}, {tk2} = N[{stoi[tk1]},{stoi[tk2]}]")
          r = stoi[tk1] # row index
          c = stoi[tk2] # col index
          N[r, c] += 1  
    
    print(N[0:15,0:15], "...")
    print("' Vic' = ", stoi[' Vic'])
    print("'ente' = ", stoi['ente'])
    print(f"N[{stoi[' Vic']}, {stoi['ente']}] =", N[stoi[' Vic'], stoi['ente']].item())
    
    # In minute 21 it shows the matrix using matplotlib