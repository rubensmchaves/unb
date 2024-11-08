import tiktoken

from util.file_utils import get_file_names

corpus_folder = "I03_bigram/corpus"

if __name__ == '__main__':
    tokenizer_name = 'cl100k_base'
    encoding = tiktoken.get_encoding(tokenizer_name)
    print(encoding.encode("I love DataCamp"))
    print(encoding.decode([40, 4048, 264, 2763, 505, 2956, 34955]))
    
    file_names = get_file_names(corpus_folder)
    for filename in file_names[:5]:
        print(filename)
    