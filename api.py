import os
from typing import Optional
import json

import chromadb
from chromadb.config import Settings
import cohere
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import torch
from transformers import BertTokenizerFast, BertModel

from translate import translate_text


def get_embeddings(query, tokenizer, model):
    query_input = tokenizer(query, return_tensors="pt", padding=False, truncation=True)
    with torch.no_grad():
        query_output = model(**query_input)
    embedding = query_output.pooler_output.tolist()[0]
    return embedding


def add_text(text, state):
    query_text = '\n'.join([x[0] + '/n' + x[1][:50] + '\n' for x in state]) + text
    print(f'{query_text=}')
    translation_response = translate_text(query_text)
    english_query_text = translation_response.translations[0].translated_text
    query_language_code = translation_response.translations[0].detected_language_code
    query_language = iso_639_1[query_language_code]
    print(f'{query_language=}')
    print(f'{english_query_text=}')
    # Get the context from chroma
    if embeddings:
        query_embeddings = get_embeddings(query_text, tokenizer, model)
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=5
        )
    else:  # Use default chromadb embeddings
        results = collection.query(
            query_texts=[english_query_text],
            n_results=5
        )

    # Prompt.
    context = '\n'.join(results['documents'][0])
    print(f'{context=}')

    # Construct prompt.
    chat_prefix = "The following is a conversation with an AI assistant for Bible translators. The assistant is"
    chat_prefix += f" helpful, creative, clever, and very friendly. The assistant only responds in the {query_language} language.\n"
    prompt = (
        chat_prefix +
        f'Read the paragraph below and answer the question, using only the information in the context, in the {query_language} language. '
        f'If the question cannot be answered based on the context alone, write "sorry i had trouble answering this question, based on the information i found\n'
        f"\n"
        f"Context:\n"
        f"{ context }\n"
        f"\n"
    )

    if len(state) > 0:
        if len(state) > 3:
            trim_state = state[-3:]
        else:
            trim_state = state
        for exchange in trim_state:
            prompt += "\nHuman: " + exchange[0] + "\nAI: " + exchange[1]
        prompt += "\nHuman: " + text + "\nAI: "
    else:
        prompt += "\nHuman: " + text + "\nAI: "
    print(f'{prompt=}')
    
    if llm == 'cohere':
        # Get the completion from co:here.
        response = co.generate(model='xlarge',
                            prompt=prompt,
                            max_tokens=200,
                            temperature=0)
        answer = response.generations[0].text

    elif llm == 'chatgpt':
        #ChatGPT reponse
        response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )
        answer = response['choices'][0]["message"]["content"]
    else:
        print("No LLM specified")
        return '', state
    
    print(f'{answer=}')

    answer = answer.split("\nHuman:")[0]
    completion = answer + " ("
    for meta in results['metadatas'][0]:
        completion = completion + meta["citation"] + "; "
    completion = completion[0:-2] + ")"

    state.append((text, answer))
    return completion, state


class TextIn(BaseModel):
    text: str
    state: Optional[list] = None


class TextOut(BaseModel):
    text: str
    state: list

app = FastAPI()


@app.post("/ask", response_model=TextOut)
def ask(input: TextIn):
    print(input)
    if input.state is None:
        input.state = []

    text, state = add_text(input.text, input.state)
    print(f'{text=}')
    print(f'{state=}')
    return {'text': text, 'state': state}

llm = 'chatgpt'
# llm = 'cohere'
embeddings = None  # Use default chromadb embeddings
# embeddings = 'labse'

# co:here setup
if llm == 'cohere':
    co = cohere.Client(os.environ["COHERE_KEY"])
elif llm == 'chatgpt':
    openai.api_key = os.environ.get("OPENAI_KEY")

if embeddings and embeddings.lower() == 'labse':
    cache_path = 'bert_cache/'
    tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE', cache_dir=cache_path)
    model = BertModel.from_pretrained('setu4993/LaBSE', cache_dir=cache_path).eval()

with open('iso639-1.json') as f:
    iso_639_1 = json.load(f)

# Vector store (assuming the .chromadb directory already exists)
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb" 
))

if embeddings and embeddings.lower() == 'labse':
    collection = client.get_collection("tyndale-labse")
else:
    collection = client.get_collection("tyndale")

