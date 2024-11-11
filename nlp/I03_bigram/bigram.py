import tiktoken
import json
import torch

from typing import List
from typing import Set
from util.file_utils import get_file_names
from util.file_utils import train_test_split

# Configuration
corpus_folder  = "corpus"
end_token      = "<|endoftext|>"
tokenizer_name = 'cl100k_base'

# Initialization
tokenizer = tiktoken.get_encoding(tokenizer_name)
bigrams_dict = {} # dictionary of bigram
vocabulary: Set[str] = None


class Bigram:
    def __init__(self, vocabulary, tokenizer, special_token):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.special_token = special_token
        self.table_freq = torch.zeros((len(vocabulary), len(vocabulary)), dtype=torch.int32) # Table of frequencies
        # Sort the vocabulary 
        sort_voc = sorted(vocabulary)        
        # Remove the especial token and add it at the begining of the sorted vocabulary
        if self.special_token:
            try:
                sort_voc.remove(self.special_token) 
                sort_voc = [self.special_token] + sort_voc        
            except:
                pass     
        # Create a index to token and a token to index mapping
        self.stoi = {s:i for i, s in enumerate(sort_voc)}  # stoi - string (word) to integer    
        self.itos = {i:s for s, i in stoi.items()}
                
                
    def encode(self, text) -> List[int]:
        return self.tokenizer.encode(text, allowed_special={self.special_token})

        
    def decode(self, encoded_tokens: List[int]) -> List[str]:
        return self.tokenizer.decode(encoded_tokens)


    def decode_single_token(self, encoded_tokens: List[int]) -> List[str]:
        return [self.decode([tk]) for tk in encoded_tokens]


    def part_trainig(self, text):
        # Add special token to mark the begining and the end of the text
        encoded_tokens = self.encode(self.special_token + text + self.special_token)
        tokens = decode_single_token(encoded_tokens)
        # Update the frequency table
        for tk1, tk2 in zip(tokens, tokens[1:]):      
            r = self.stoi[tk1] # row index
            c = self.stoi[tk2] # col index
            self.table_freq[r, c] += 1



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


def main():
    # Get file names from a folder ('corpus') and separate it into traning set and test set.
    file_names = sorted(get_file_names(corpus_folder))
    print("Test function 'train_test_split':")
    train_set, test_set = train_test_split(file_names, test_size=0.2)
    n_samples = 5
    print(f"Files set (samples): {file_names[:n_samples]}... ({n_samples} of {len(file_names)})")
    print(f"Train Set (samples): {train_set[:n_samples]}... ({n_samples} of {len(train_set)})")
    print(f"Test Set (samples): {test_set[:n_samples]}... ({n_samples} of {len(test_set)})")    
    
    # Load files and store its content ('text' attribute) into a list of texts
    texts = []
    for filename in train_set:  # TODO: Substituir 'file_names' por 'train_set'
        with open(f"{corpus_folder}/{filename}", "r", encoding='utf-8') as file:
            print(f"loading... ({filename})")
            data = json.load(file);
            text = data.get("text", "")
            texts.append(end_token + text + end_token)  # Append text and add space


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
    vocabulary = None
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
    print('Sorted bigrams frenquecies (descending):')  
    bigram_list = sorted(bigrams_dict.items(), key = lambda value: value[1], reverse=True)
    print(bigram_list[:5], '...')   
    tkn_freq = decode_bigram_freq(bigrams_dict)
    tkn_freq = sorted(tkn_freq.items(), key = lambda value: value[1], reverse=True)
    print(tkn_freq[:5], '...', '\n')   
    bigram_tk_A = tkn_freq[0][0][0]
    bigram_tk_B = tkn_freq[0][0][1]

    # Maps it token (string) to a integer (sequencialy). For simplification we make the 'end_token' be the first element of the dictionaries ('stoi' and 'itos')
    sort_voc = sorted(vocabulary)
    sort_voc.remove(end_token)
    print(f"Is '{end_token}' into the Vocabulary? {end_token in sort_voc} ({sort_voc[:5]} ...)")
    sort_voc = [end_token] + sort_voc
    print(f"Is '{end_token}' into the Vocabulary? {end_token in sort_voc} ({sort_voc[:5]} ...)")
    stoi = {s:i for i, s in enumerate(sort_voc)}  # stoi - string (word) to integer    
    itos = {i:s for s, i in stoi.items()}
    print("Dicionary: 'stoi'")
    print(list(stoi.items())[:10], '...')
    print(f"most frequency bigram (index): ({stoi[bigram_tk_A]}, {stoi[bigram_tk_B]})")
    print("Dicionary: 'itos'")
    print(list(itos.items())[:10], '...')
    print(f"most frequency bigram (str): ('{itos[stoi[bigram_tk_A]]}', '{itos[stoi[bigram_tk_B]]}')")
    print(f"Vocabulary size: {len(sort_voc)}")
    print(f"stoi: {len(stoi)}")
    print(f"itos: {len(itos)}", "\n")
    
    # Create table of frequencies for bigrams
    print("Frequency table:")
    total_tokens = len(stoi)
    N = torch.zeros((total_tokens, total_tokens), dtype=torch.int32)
    for text_tkn in texts_tokens:
        for tk1, tk2 in zip(text_tkn, text_tkn[1:]):      
          r = stoi[tk1] # row index
          c = stoi[tk2] # col index
          N[r, c] += 1  
    
    print(N[0:15,0:15], "...")
    print(f"'{bigram_tk_A}' = ", stoi[bigram_tk_A])
    print(f"'{bigram_tk_B}' = ", stoi[bigram_tk_B])
    print(f"N[{stoi[bigram_tk_A]}, {stoi[bigram_tk_B]}] =", N[stoi[bigram_tk_A], stoi[bigram_tk_B]].item(), "\n")
    # In minute 21 it shows the matrix using matplotlib

    # Generate the probability matrix
    print(N[0].shape)
    p = N[0].float()
    print(p[:20])
    print(f"Sum() = {p.sum()}")
    p = p / p.sum()
    print(p[:20])    
    
    
    print("\nTest text generation...")
    text_frag = "São Vicente e Ventosa é uma freguesia"
    encoded_frag = encode(text_frag)
    frag_tokens = decode_single_token(encoded_frag)    
    idx = stoi[frag_tokens[-1]]
    print(f"Fragmento de texto: '{text_frag}'")
    print(f"Tokens: {frag_tokens}")
    print(f"Last token: '{itos[idx]}'")

    # Text generation.
    seed = torch.Generator().manual_seed(2147483647) # Tensor genetator
    new_text = ''
    while True:
        p = N[idx].float()
        p = p / p.sum()
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=seed).item()
        new_text += itos[idx]
        if idx == 0:
            break
    
    print(new_text)
    
    #20:55 - fala sobre a possibilidade de uma linha ter apenas zeros na tabela (matriz) de frequência
    
    #22:33 - simplifica o caracter de final de palavra (text) 
    
    #33:02


if __name__ == '__main__':
    main()