import tiktoken

tokenizer_name = 'cl100k_base'
encoding = tiktoken.get_encoding(tokenizer_name)
print(encoding.encode("I love DataCamp"))
print(encoding.decode([40, 4048, 264, 2763, 505, 2956, 34955]))