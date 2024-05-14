# STEP:1 
# NOTE: the text files have been edited manually, do not run this code again
# the outputs are saved in processed_text_index folder as all_texts.txt and {label}_all_texts.txt

# codes to process the texts folder. To get all the text prompts in single files 

import os


label_dict = {0: ["walk"], 
              1: ["run","ran","sprint","jog"],
              2: ["kick", "box" "fight", "punch", "cartwheel","handstand", "flip", "jump", "spin", "play", "dance", "climb", "swim", "push", "crawl", "shove", "lie", "ride", "hit", "shout", "kneel"],
            }

def processAllTexts(txtpath="texts"):
    Lines = []
    for file in os.listdir(txtpath):
        if file.endswith(".txt"):
            # read the file
            with open(os.path.join(txtpath, file), 'r') as f:
                text = f.read()
                text = text.lower()
                text = text.split('#')[0]
                # replace all the punctuation with spaces
                text = text.replace('.', ' ')
                text = text.replace(',', ' ')
                text = text.replace('!', ' ')
                text = text.replace('?', ' ')
                text = text.replace(';', ' ')
                text = text.replace(':', ' ')
                Lines.append(text)
                Lines.append('\n')
    with open(f"processed_text_index/all_texts.txt", 'w') as f:
        f.writelines(Lines)

def processTexts(txtpath="texts", label=0):
    # create a single text file with all the prompts 
    Lines = []
    for file in os.listdir(txtpath):
        if file.endswith(".txt"):
            # read the file
            with open(os.path.join(txtpath, file), 'r') as f:
                text = f.read()
                text = text.lower()
                text = text.split('#')[0]
                # replace all the punctuation with spaces
                text = text.replace('.', ' ')
                text = text.replace(',', ' ')
                text = text.replace('!', ' ')
                text = text.replace('?', ' ')
                text = text.replace(';', ' ')
                text = text.replace(':', ' ')
                # if text has the label, append it to Lines
                for l in label_dict[label]:
                    if l in text:
                        Lines.append(text)
                        Lines.append('\n')
                        # to ensure that the text is not appended twice
                        break

    # save the Lines in a single file
    with open(f"processed_text_index/{label}_all_texts.txt", 'w') as f:
        f.writelines(Lines)

if __name__ == "__main__":
    processAllTexts()
    processTexts(txtpath="texts", label=2)
        

