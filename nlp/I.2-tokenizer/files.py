import os
import json
import tokenizer as tk


def verify_text_tokenization(text, verbose=False):
    tkns = tk.get_tokens(text)
    t, m = tk.tokenize(vocab_size=276, encoding_size=256, char_codes=tkns)
    text2 = tk.detokenize(t, m, 256)
    if text == text2:
        return True
    else:
        return False


def concatenate_texts_from_json(directory):
    full_text = ""
    count = 0

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                print(f"loading... ({filename})", end="")
                data = json.load(file);
                text = data.get("text", "")
                tokenizable = verify_text_tokenization(text)
                print(" tokenizable?", tokenizable)
                if not tokenizable:
                    raise Exception("Não foi possível decodificar o texto (tokenizable)!")
                full_text += text + "\n\n"  # Append text and add space
                count = count + 1
        # if count == 6:
        #     break

    print(f"{count} files loaded.")
    return full_text.strip()  # Remove trailing whitespace


if __name__ == "__main__":
    # Usage
    directory_path = "corpus"  # Replace with your directory path
    combined_text = concatenate_texts_from_json(directory_path)
    #print(combined_text)
