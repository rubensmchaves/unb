This bigram was based on instructions given by [Andrej Karpathy](https://en.wikipedia.org/wiki/Andrej_Karpathy) in the Youtube video called [_The spelled-out intro to language modeling: building makemore_]([https://www.youtube.com/watch?v=zduSFxRajkE](https://www.youtube.com/watch?v=PaCmpygFfXo)).

## Running
To run this example you have to:
1. Clone this project: 
   ```bash
   git clone https://github.com/rubensmchaves/unb.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a folder <code>corpus</code> into <code>nlp</code> folder;
5. Unzip the file [corpus.zip](https://github.com/rubensmchaves/unb/blob/main/nlp/corpus.zip) into the folder <code>nlp\corpus</code> (just created);
6. In the shell, at the folder <code>nlp</code>, execute:
   ```bash
   py -m I03_bigram.bigram
   ```
   
## References
### Video
[The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) - Andrej Karpathy

[Evaluation and Perplexity](https://www.youtube.com/watch?v=B_2bntDYano) - Daniel Jurafsky

### Text
[Python Library for Tokenizing Text](https://www.datacamp.com/tutorial/tiktoken-library-python)

[How to count tokens with Tiktoken](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) - Ted Sanders

### Github
[Makemore](https://github.com/karpathy/makemore)
