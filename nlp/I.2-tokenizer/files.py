import os
import json

def concatenate_texts_from_json(directory):
    full_text = ""
    count = 0

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                print(f"loading... ({filename})")
                data = json.load(file);
                full_text += data.get("text", "") + "\n\n"  # Append text and add space
                count = count + 1
        if count == 6:
            break

    print(f"{count} files loaded.")
    return full_text.strip()  # Remove trailing whitespace


if __name__ == "__main__":
    # Usage
    directory_path = "corpus"  # Replace with your directory path
    combined_text = concatenate_texts_from_json(directory_path)
    print(combined_text)
