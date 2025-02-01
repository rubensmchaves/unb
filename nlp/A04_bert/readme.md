This is the fourth assingment of the NLP course at UnB, fully discrebed [here](https://github.com/thiagodepaulo/nlp/blob/main/aula_10/exercicio10.md). The idea is to implement a notebook for text classification using BERT (Bidirectional Encoder Representations from Transformers) and the same _corpus_ from the previous assignemnt ([A03_text_classification](https://github.com/rubensmchaves/unb/tree/main/nlp/A03_text_classifier)).

To train and test them we may use any of the _corpus_ available [here](https://github.com/ragero/text-collections/tree/master/complete_texts_csvs).

# Setting up

## Virtual Environment

To run the project is necessary to create a virtual enviroment to avoid libraries conflits. So, we created a virtual environment using `venv` because it comes with Python 3.4+. For more information about it click [here](https://python.land/virtual-environments/virtualenv).

After start the virtual environment it is important to install the required libraries, try this (for more [here](https://www.geeksforgeeks.org/install-packages-using-pip-with-requirements-txt-file-in-python/)): 
```bash
   py -m pip install -r requirements.txt
```

## References

### Video

[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Andrej Karpathy

#### [Neural Network Course](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - 3Blue1Brown

1. [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)

2. [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)

3. [Backpropagation, step-by-step](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

4. [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)

5. [Transformers (how LLMs work) explained visually](https://www.youtube.com/watch?v=wjZofJX0v4M)

6. [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc)

7. [How might LLMs store facts](https://www.youtube.com/watch?v=9-Jl0dxWQs8)
  

### Text
[Mastering BERT Model: Building it from Scratch with Pytorch](https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891)

[Mastering BERT: A Comprehensive Guide from Beginner to Advanced in Natural Language Processing (NLP)](https://medium.com/@shaikhrayyan123/a-comprehensive-guide-to-understanding-bert-from-beginners-to-advanced-2379699e2b51) - not so good

[Text Classification Using Hugging Face(Fine-Tuning)](https://medium.com/@sandeep.ai/text-classification-using-hugging-face-fine-tuning-43c7416b049b) 

[HuggingFace: NLP Course](https://huggingface.co/learn/nlp-course/en/chapter0/1?fw=pt)

[How to Finetune BERT for Text Classification (HuggingFace Transformers, Tensorflow 2.0) on a Custom Dataset](https://victordibia.com/blog/text-classification-hf-tf2/)

[HuggingFace (docs): Transformers](https://huggingface.co/docs/transformers/index) ([text classification](https://huggingface.co/docs/transformers/tasks/sequence_classification))
