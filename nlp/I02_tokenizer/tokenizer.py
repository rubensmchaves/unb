from typing import List

def get_tokens(text: str, verbose=False):
    tokens = text.encode("utf-8") # byte sequence
    if verbose:
        print("text:", tokens)
        print("length:", len(text))
        print("---")
    tokens = list(map(int, tokens)) # create a list of integers (0..255)
    if verbose:
        print("tokens:", tokens)
        print("length:", len(tokens))
        print("---")
    return tokens


def get_frequencies(char_codes: List[int]):
    """
    Calculates the frequency of each consecutive pair of integers in a list.

    Args:
        char_codes (List[int]): A list of integer codes to analyze for consecutive pair frequencies.

    Returns:
        Dict[Tuple[int, int], int]: A dictionary where each key is a tuple representing a consecutive pair
                                    of integers from `char_codes`, and each value is the frequency of that pair.

    Example:
        >>> get_frequencies([1, 2, 2, 3, 2, 2])
        {(1, 2): 1, (2, 2): 2, (2, 3): 1, (3, 2): 1}
    """
    counts = {} # dictionary initialization
    # zip: combine multiple iterables (like lists, tuples, etc.) element by element
    # char_codes: is an iterable
    # char_codes[1:]: the iterable shifted one position (to combine the consecutive element)
    for pair in zip(char_codes, char_codes[1:]):
        # counts.get(pair, 0): gets the value (frequency) of the pair or return zero if it does not exist
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(char_codes, pair, new_code):
    """
    Merges consecutive occurrences of a specified pair of integers in a list,
    replacing each occurrence with a single new integer code.

    Args:
        char_codes (List[int]): A list of integer codes to be processed.
        pair (Tuple[int, int]): A tuple containing two integers representing the pair to merge.
        new_code (int): The integer code that will replace each occurrence of the pair in the list.

    Returns:
        List[int]: A new list where each occurrence of `pair` is replaced by `new_code`.
                   Elements that do not match `pair` remain unchanged.

    Example:
        >>> merge([1, 2, 2, 3, 2, 2], (2, 3), 99)
        [1, 2, 99, 2, 2]
    """
    new_codes = []
    i = 0
    while i < len(char_codes):
        # if we are not at the very last position AND the pair matches, replace it
        # 'i < len(char_codes)-1' is used to avoid index out of bound error because of 'char_codes[i+1]'
        if i < len(char_codes)-1 and char_codes[i] == pair[0] and char_codes[i+1] == pair[1]:
            new_codes.append(new_code)
            i += 2
        else:
            new_codes.append(char_codes[i])
            i += 1
    return new_codes


def tokenize(char_codes: List[int], vocab_size: int, encoding_size=256, verbose=False):
    """
    Tokenizes a list of integer codes by merging frequently occurring pairs until a specified vocabulary size is achieved.

    Args:
        char_codes (List[int]): A list of integer codes representing the initial tokens to be processed.
        vocab_size (int): The target vocabulary size after merging.
        encoding_size (int): The initial encoding size, which is the starting number of unique tokens.
        verbose (bool, optional): If True, prints detailed output of the merging process. Defaults to False.

    Returns:
        Tuple[List[int], Dict[Tuple[int, int], int]]:
            - A list of integer codes after merging, representing the tokenized result.
            - A dictionary where each key is a tuple representing a merged pair of tokens, and each value is the new token integer assigned to that pair.

    Example:
        >>> tokenize(10, 5, [1, 2, 2, 3, 2, 2], verbose=True)
        merging (2, 2) into new token 5
        merging (2, 3) into new token 6
        tokens length: 6
        codes length: 4
        compression ratio: 1.50X
        ---
        ([1, 5, 6, 5], {(2, 2): 5, (2, 3): 6})

    Notes:
        - The function works by identifying the most frequent consecutive pairs of codes, replacing them with a new token code,
          and continuing the process until the target vocabulary size is reached.
        - This is useful for tasks where a smaller, compressed token representation is needed.
    """
    codes = list(char_codes) # copy so we do not destroy the original list
    num_merges = vocab_size - encoding_size
    merges = {} # {int, int} -> int
    for i in range(num_merges):
        if len(codes) > 1:
            # TODO: At this point the code could be improved by avoid generating new frequencies to the whole vector, it should compute the frequency only for the elements beside the merged elements.
            freq_distribution = get_frequencies(codes)
            pair = max(freq_distribution, key=freq_distribution.get)
            new_code = encoding_size + i
            if verbose:
                print(f"merging {pair} into new token {new_code}")
            codes = merge(codes, pair, new_code)
            merges[pair] = new_code
        else:
            break

    if verbose:
        print("tokens length:", len(char_codes))
        print("codes length:", len(codes))
        print(f"compression ratio: {len(char_codes) / len(codes):.2f}X")
        print("---")

    return codes, merges


def get_vocabulary(encoding_mapping, encoding_size=256):
    """
    Generates a vocabulary dictionary that maps encoding indices to byte sequences.

    This function constructs a vocabulary dictionary, where each index (up to `encoding_size`)
    initially maps to its corresponding byte representation. For each pair in the `encoding_mapping`,
    the function merges two byte sequences to create new tokens, expanding the vocabulary.

    Parameters:
    ----------
    encoding_mapping : dict
        A dictionary where keys are tuples (p0, p1) representing pairs of existing indices
        in the vocabulary, and values are new indices assigned to merged byte sequences.

    encoding_size : int, optional
        The initial size of the encoding, representing the number of unique byte values.
        Default is 256, covering the standard byte range.

    Returns:
    -------
    dict
        A vocabulary dictionary where each key is an index, and each value is a byte sequence
        (of type `bytes`) representing the token mapped to that index.

    Example:
    --------
    >>> encoding_mapping = {(65, 66): 256, (256, 67): 257}
    >>> vocab = get_vocabulary(encoding_mapping)
    >>> vocab[256]  # Corresponds to the merged bytes for indices 65 and 66
    b'AB'
    """
    vocab = {idx: bytes([idx]) for idx in range(encoding_size)}
    for (p0, p1), idx in encoding_mapping.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab


def detokenize(token_codes, encoding_mapping, encoding_size=256):
    """
    Decodes a list of token codes into a UTF-8 string using a predefined vocabulary and encoding mapping.

    Args:
        token_codes (List[int]): A list of integer codes representing the encoded tokens.
        encoding_mapping (Dict[Tuple[int, int], int]): A dictionary that maps pairs of tokens (tuples of integers)
                                                       to new token indices created during encoding.
        encoding_size (int): The initial encoding size, representing the number of base tokens (0 to encoding_size-1).

    Returns:
        str: A decoded UTF-8 string constructed from the token codes.

    Example:
        >>> encoding_mapping = {(65, 66): 256, (256, 67): 257}
        >>> decoder([65, 257, 67, 256], encoding_mapping, 256)
        'ABC'

    Notes:
        - The function first constructs a vocabulary with base tokens (from 0 to encoding_size-1) as single-byte values.
        - It then iteratively builds larger tokens using `encoding_mapping`.
        - The token codes are combined into a single byte string, which is then decoded to UTF-8 with error handling for
          invalid sequences.
    """
    vocab = get_vocabulary(encoding_mapping, encoding_size)
    tokens = b"".join(vocab[idx] for idx in token_codes)
    text = tokens.decode("utf-8", errors="replace")
    return text


if __name__ == "__main__":
    # https://pt.wikipedia.org/wiki/Jesus
    wiki_text = "Jesus, também chamado Jesus de Nazaré (n. 7–2 a.C. – m. 30–33 d.C.) foi um pregador e líder religioso judeu do primeiro século.[11] É a figura central do cristianismo e aquele que os ensinamentos de maior parte das denominações cristãs, além dos judeus messiânicos, consideram ser o Filho de Deus. O cristianismo e o judaísmo messiânico consideram Jesus como o Messias aguardado no Antigo Testamento e referem-se a ele como Jesus Cristo, um nome também usado fora do contexto cristão."
    tkns = get_tokens(wiki_text, False)
    t, m = tokenize(vocab_size=300, char_codes=tkns, verbose=True)
    print(t)
    print(m)

    #  Show vocabulary
    vocab = get_vocabulary(m)
    for value in vocab.values():
        print(f"'{value.decode('utf-8', errors='replace')}'")

    #testing (verification)
    text2 = detokenize(t, m, 256)
    if wiki_text == text2:
        print(True)
    else:
        print(False)