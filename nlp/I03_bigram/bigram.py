import tiktoken
import json
import torch
import math

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


def log(*messages, verbose=True):
    """
    Logs messages to the console if verbose is True.

    Parameters:
    - messages (str): Messages to be logged.
    - verbose (bool): Flag to control logging. Defaults to True.

    Returns:
    - None
    """
    if verbose:
        message = ''
        for msg in messages:
            message = ' ' + msg
        print(message)


def encode(text) -> List[int]:
    """
    Encodes a given text into a list of token IDs.

    Parameters:
    - text (str): The text to encode.

    Returns:
    - List[int]: A list of token IDs representing the text.
    """
    return tokenizer.encode(text, allowed_special={end_token})


def decode(tokens: List[int]) -> List[str]:
    """
    Decodes a list of token IDs into text.

    Parameters:
    - tokens (List[int]): The list of token IDs to decode.

    Returns:
    - List[str]: The decoded text.
    """
    return tokenizer.decode(tokens)


def decode_single_token(tokens: List[int]) -> List[str]:
    """
    Decodes each token in a list of tokens individually.

    Parameters:
    - tokens (List[int]): A list of token IDs to decode.

    Returns:
    - List[str]: A list of decoded single tokens as strings.
    """
    return [decode([tk]) for tk in tokens]


def compute_bigram_frequency(encoded_txt):
    """
    Computes the frequency of each bigram in the encoded text.

    Parameters:
    - encoded_txt (List[int]): A list of token IDs representing encoded text.

    Returns:
    - dict: A dictionary with bigram tuples as keys and their frequencies as values.
    """
    for tk1, tk2 in zip(encoded_txt, encoded_txt[1:]):
        b = (tk1, tk2)
        bigrams_dict[b] = bigrams_dict.get(b, 0) + 1  
    return bigrams_dict


def decode_bigram_freq(bigrams_dict):
    """
    Decodes each bigram in a frequency dictionary to text.

    Parameters:
    - bigrams_dict (dict): A dictionary with bigram tuples as keys and their frequencies as values.

    Returns:
    - dict: A dictionary with decoded bigrams as keys and their frequencies as values, or None if the input dictionary is empty.
    """
    if len(bigrams_dict) > 0:
        decode_bigrams_dict = {}
        for key, value in bigrams_dict.items():
            decoded_key = (decode_bigram(key))
            decode_bigrams_dict[decoded_key] = value
        return decode_bigrams_dict
    else:
        return None


def decode_bigram(encoded_bigram: Set[int]) -> Set[str]:
    """
    Decodes a single encoded bigram to text.

    Parameters:
    - encoded_bigram (Set[int]): A set containing two token IDs representing a bigram.

    Returns:
    - Set[str]: A tuple of two decoded tokens as strings, or None if the input is empty.
    """
    if encoded_bigram:
        bigrams = list(encoded_bigram)
        b1 = decode([bigrams[0]])
        b2 = decode([bigrams[1]])
        return (b1, b2)
    else:
        return None


def decode_bigrams(encoded_bigrams: List[Set[int]]) -> List[Set[str]]:
    """
    Decodes a list of encoded bigrams to text.

    Parameters:
    - encoded_bigrams (List[Set[int]]): A list of sets containing token IDs representing bigrams.

    Returns:
    - List[Set[str]]: A list of tuples with decoded bigrams as strings, or None if the input list is empty.
    """
    if len(encoded_bigrams) > 0:
        decode_bigrams_list = []
        for b in encoded_bigrams:
            decode_bigrams_list.append(decode_bigram(b))
        return decode_bigrams_list
    else:
        return None


def compute_perplexity(encoded_text: List[str], table_probabilities, stoi_mapping):
    """
    Computes the perplexity of a text using bigram probabilities.

    Parameters:
    - encoded_text (List[str]): A list of tokens representing the encoded text.
    - table_probabilities (2D array): A 2D array representing the bigram probabilities.
    - stoi_mapping (dict): A dictionary mapping tokens to their indices.

    Returns:
    - float: The perplexity of the encoded text, or None if the input is empty.
    """
    if encoded_text:
        N = len(encoded_text)
        log_prob_sum = 0.0
        for i in range(1, N):
            tk0 = encoded_text[i-1]
            tk1 = encoded_text[i]
            if (tk0 in stoi_mapping and tk1 in stoi_mapping):
                i = stoi_mapping[tk0]
                j = stoi_mapping[tk1]
                prob = table_probabilities[i, j]
                log(f"('{tk0}', '{tk1}'): {prob:.4f}")
            else:
                prob = 1e-10
                log(f"('{tk0}', '{tk1}'): 1e-10")
            log_prob_sum += math.log(prob)
        return math.exp(-log_prob_sum / N)
    else:
        return None


def text_generation(last_token, table_probabilities, stoi_mapping, itos_mapping):
    """
    Generates text using bigram probabilities starting from a given token.

    Parameters:
    - last_token (str): The initial token to start generation.
    - table_probabilities (2D array): A 2D array representing the bigram probabilities.
    - stoi_mapping (dict): A dictionary mapping tokens to their indices.
    - itos_mapping (dict): A dictionary mapping indices to their tokens.

    Returns:
    - str: The generated text.
    """
    seed = torch.Generator().manual_seed(2147483647) # Tensor genetator
    new_text = ''
    idx = stoi_mapping[last_token]
    while True: #True:
        p = table_probabilities[idx]
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=seed).item()
        new_text += itos_mapping[idx]
        if idx == 0:
            break
    return new_text
    

def main():
    # Get file names from a folder ('corpus') and separate it into traning set and test set.
    file_names = sorted(get_file_names(corpus_folder))
    log("Test function 'train_test_split':")
    train_set, test_set = train_test_split(file_names, test_size=0.2)
    n_samples = 5
    log(f"Files set (samples): {file_names[:n_samples]}... ({n_samples} of {len(file_names)})")
    log(f"Train Set (samples): {train_set[:n_samples]}... ({n_samples} of {len(train_set)})")
    log(f"Test Set (samples): {test_set[:n_samples]}... ({n_samples} of {len(test_set)})")    
    
    # Load files and store its content ('text' attribute) into a list of texts
    texts = []
    for filename in file_names[:1]:  # TODO: Substituir 'file_names' por 'train_set'
        with open(f"{corpus_folder}/{filename}", "r", encoding='utf-8') as file:
            log(f"loading... ({filename})")
            data = json.load(file);
            text = data.get("text", "")
            texts.append(end_token + text + end_token)  # Append text and add space


    # Total of tokens
    cod_tokens = encode(texts[0])
    log(f"Total of tokens: {len(cod_tokens)}")  
    # Show text encode and decode
    log(texts[0][:80])
    log(f"{cod_tokens[:20]} ...")
    log(f"{decode_single_token(cod_tokens)[:10]} ...")
    log(f"decode(cod_tokens[:20]) ...\n")

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
    log(f"Vocalubary size: {len(vocabulary)}")
    log('Bigrams:')
    log(f"{list(bigrams_dict.keys())[:5]} ...")  
    decoded_bigrams_list = decode_bigrams(list(bigrams_dict.keys()))
    log(f"{list(decoded_bigrams_list)[:5]} ...")

    # Show part of the bigrams       
    log('Bigrams frenquecies:')  
    bigram_list = list(bigrams_dict.items())
    log(f"{bigram_list[:5]} ...")
    tkn_freq = decode_bigram_freq(bigrams_dict)
    tkn_freq = list(tkn_freq.items())
    log(f"{tkn_freq[:5]} ...")   

    # Sorted bigrams by frequency
    log('Sorted bigrams frenquecies (descending):')  
    bigram_list = sorted(bigrams_dict.items(), key = lambda value: value[1], reverse=True)
    log(f"{bigram_list[:5]} ...")   
    tkn_freq = decode_bigram_freq(bigrams_dict)
    tkn_freq = sorted(tkn_freq.items(), key = lambda value: value[1], reverse=True)
    log(f"{tkn_freq[:5]} ...\n")   
    bigram_tk_A = tkn_freq[0][0][0]
    bigram_tk_B = tkn_freq[0][0][1]

    # Maps it token (string) to a integer (sequencialy). For simplification we make the 'end_token' be the first element of the dictionaries ('stoi' and 'itos')
    sort_voc = sorted(vocabulary)
    sort_voc.remove(end_token)
    log(f"Is '{end_token}' into the Vocabulary? {end_token in sort_voc} ({sort_voc[:5]} ...)")
    sort_voc = [end_token] + sort_voc
    log(f"Is '{end_token}' into the Vocabulary? {end_token in sort_voc} ({sort_voc[:5]} ...)")
    stoi = {s:i for i, s in enumerate(sort_voc)}  # stoi - string (word) to integer    
    itos = {i:s for s, i in stoi.items()}
    log("Dicionary: 'stoi'")
    log(f"list(stoi.items())[:10] ...")
    log(f"most frequency bigram (index): ({stoi[bigram_tk_A]}, {stoi[bigram_tk_B]})")
    log("Dicionary: 'itos'")
    log(f"{list(itos.items())[:10]} ...")
    log(f"most frequency bigram (str): ('{itos[stoi[bigram_tk_A]]}', '{itos[stoi[bigram_tk_B]]}')")
    log(f"Vocabulary size: {len(sort_voc)}")
    log(f"stoi: {len(stoi)}")
    log(f"itos: {len(itos)}", "\n")
    
    # Create table of frequencies for bigrams
    log("Frequency table:")
    total_tokens = len(stoi)
    N = torch.zeros((total_tokens, total_tokens), dtype=torch.int32)
    for text_tkn in texts_tokens:
        for tk1, tk2 in zip(text_tkn, text_tkn[1:]):      
          r = stoi[tk1] # row index
          c = stoi[tk2] # col index
          N[r, c] += 1  
    
    log(f"N[0:15,0:15] ...")
    log(f"'{bigram_tk_A}' = {stoi[bigram_tk_A]}")
    log(f"'{bigram_tk_B}' = {stoi[bigram_tk_B]}")
    log(f"N[{stoi[bigram_tk_A]}, {stoi[bigram_tk_B]}] = {N[stoi[bigram_tk_A], stoi[bigram_tk_B]].item()} \n")
    # In minute 21 it shows the matrix using matplotlib

    # Compute the table of probabilities
    table_probabilities = (N+1).float()
    table_probabilities /= table_probabilities.sum(1, keepdim=True)

    # Compute preplexity
    #text = "Vicente e Ventosa se conheceram no município de Elvas"
    text = "Eu me chamo Rubens Marques Chaves e tenho 44 anos de idade"
    encoded_text = decode_single_token(encode(text))
    perplexity = compute_perplexity(encoded_text, table_probabilities, stoi)
    log(f"Perplexity: {perplexity}")

    # Testing text generation...
    log("\nTest text generation...")
    text_frag = "São Vicente e Ventosa é uma freguesia"
    encoded_frag = encode(text_frag)
    frag_tokens = decode_single_token(encoded_frag)    
    last_token = frag_tokens[-1]
    log(f"Tokens: {frag_tokens}")
    log(f"Last token: '{last_token}'")
    log(f"Text fragment: {text_frag} (...)")
    new_text = text_generation(last_token, table_probabilities, stoi, itos)
    log("Text completion: \n(...)", new_text)
    

if __name__ == '__main__':
    main()
    