'''Import statments'''
import os
import re

import pandas as pd
import streamlit as st
import tiktoken
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from spacy.cli.benchmark_speed import count_tokens
from transformers import T5ForConditionalGeneration, T5Tokenizer

'''Proxy settting'''
os.environ["HTTP_PROXY"] = "proxy.its.hpecorp.net:8080"
os.environ["HTTPS_PROXY"] = "proxy.its.hpecorp.net:8080"

'''load env'''
load_dotenv()

'''Api Key'''
cohere_api_key = API Key
os.environ['COHERE_API_KEY'] = cohere_api_key


def read_pdf(uploaded_file):
    '''
    :param uploaded_file: take the upload function
    :return: filtered tokens and messages
    '''
    try:
        with uploaded_file as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        tokens = word_tokenize(text)
        unwanted_patterns = [r'\d+', r'\w*\d\w*']
        filtered_tokens = [token for token in tokens if
                           not any(re.match(pattern, token) for pattern in unwanted_patterns)]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text, "PDF file read successfully!"
    except Exception as e:
        return None, f"An error occurred while reading the PDF file: {e}"


def read_file(uploaded_file):
    '''
    :param uploaded_file: Read excel and pdf file
    :return: text and message
    '''
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension == '.csv':
        df = pd.read_csv(uploaded_file)
        return df, "CSV file read successfully!"
    elif file_extension == '.xlsx':
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        return df, "Excel file read successfully!"
    elif file_extension == '.txt':
        file_content = uploaded_file.read().decode('utf-8')
        lines = file_content.splitlines()
        split_data = [line.split('|') for line in lines]
        split_data = [[item.strip() for item in line] for line in split_data]
        data = split_data[1:]
        df = pd.DataFrame(data)
        return df, "Text file read successfully!"
    elif file_extension == '.pdf':
        text, message = read_pdf(uploaded_file)
        return text, "PDF file read successfully!"
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}. Please upload a CSV, Excel, or PDF file.")


def generate_answer(question, context):
    '''
    :param question: take the question
    :param context: take the context
    :return: generate the answer
    '''

    # Initialize the T5 model and tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Prepare input
    input_text = f"question: {question} context: {context}"
    inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the answer
    outputs = t5_model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=50)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def truncate_context(question, context, max_tokens):
    '''
    :param question: user question
    :param context:  context
    :param max_tokens:  max tokes
    :return: trucated valesu
    '''
    tokenizer = tiktoken.get_encoding('gpt2')
    question_tokens = count_tokens(question)
    context_tokens = count_tokens(context)
    prompt_tokens = count_tokens(f"Question: \nContext: \nAnswer:")
    total_tokens = question_tokens + context_tokens + prompt_tokens

    if total_tokens > max_tokens:
        excess_tokens = total_tokens - max_tokens
        context_token_ids = tokenizer.encode(context)
        truncated_context_token_ids = context_token_ids[:-excess_tokens]
        truncated_context = tokenizer.decode(truncated_context_token_ids)
    else:
        truncated_context = context

    return truncated_context


def main():
    st.header("Project with AI")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        try:
            data, message = read_file(uploaded_file)

            st.success(message)
            user_question = st.text_input("Ask a question:")
            if user_question:
                context = data.to_string() if isinstance(data, pd.DataFrame) else data
                response = generate_answer(user_question, context)
                st.write("AI Answer:", response)
        except ValueError as e:
            st.error(e)


if __name__ == "__main__":
    main()
