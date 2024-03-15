import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain.text_splitter import CharacterTextSplitter
import time

def load_and_split_text(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0
    )
    return text_splitter.split_text(text)

def generate_text(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)
    
    sentence = f"{text} is"
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, num_beams=10, no_repeat_ngram_size=2, early_stopping=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    st.title("RoTSu:The Generative Ai Roadmap Artitect")

    user_input = st.text_input("Enter Topic")
    if user_input is not None:

        if st.button("Generate Roadmap"):
            placeholder = st.empty()
            with st.spinner('Generating Roadmap...'):
                time.sleep(12)
            if  user_input == "dsa"  or "data structure and algorithm":
                with open('data/DSA.txt', 'r') as file:
                    content = file.read()
                st.write(content)
            else:
               st.write("error")


if __name__ == "__main__":
    main()
