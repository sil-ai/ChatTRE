import os

import gradio as gr
import chromadb
from chromadb.config import Settings
import cohere


# co:here setup
co = cohere.Client(os.environ["COHERE_KEY"])

# Vector store (assuming the .chromadb directory already exists)
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb" 
))
collection = client.get_collection("tyndale")

# Prompt reference.
chat_prefix = "The following is a conversation with an AI assistant for Bible translators. The assistant is"
chat_prefix += " helpful, creative, clever, and very friendly.\n"


# QuestionID provides some help in determining if a sentence is a question.
class QuestionID:
    """
        QuestionID has the actual logic used to determine if sentence is a question
    """
    def padCharacter(self, character: str, sentence: str):
        if character in sentence:
            position = sentence.index(character)
            if position > 0 and position < len(sentence):

                # Check for existing white space before the special character.
                if (sentence[position - 1]) != " ":
                    sentence = sentence.replace(character, (" " + character))

        return sentence

    def predict(self, sentence: str):
        questionStarters = [
            "which", "wont", "cant", "isnt", "arent", "is", "do", "does",
            "will", "can"
        ]
        questionElements = [
            "who", "what", "when", "where", "why", "how", "sup", "?"
        ]

        sentence = sentence.lower()
        sentence = sentence.replace("\'", "")
        sentence = self.padCharacter('?', sentence)
        splitWords = sentence.split()

        if any(word == splitWords[0] for word in questionStarters) or any(
                word in splitWords for word in questionElements):
            return True
        else:
            return False
        

# Question detection setup.
model = QuestionID()


def add_text(state, text):

    # Determine if the text is a question.
    isQuestion = model.predict(text)
    if isQuestion:

        # Get the context from chroma
        results = collection.query(
            query_texts=[text],
            n_results=3
        )

        # Prompt.
        context = results['documents'][0][0]
        qa_prompt = (
            f'read the paragraph below and answer the question, if the question cannot be answered based on the context alone, write "sorry i had trouble answering this question, based on the information i found\n'
            f"\n"
            f"Context:\n"
            f"{ context }\n"
            f"\n"
            f"Question: { text }\n"
            "Answer:")

        # Answer the question
        response = co.generate(model='xlarge',
                            prompt=qa_prompt,
                            max_tokens=100,
                            temperature=0.3)
        
        # format the answer
        completion = response.generations[0].text.split("Context:")[0] + "("
        for meta in results['metadatas'][0]:
            completion = completion + meta["citation"] + "; "
        completion = completion[0:-2] + ")"
    
    else:

        # Construct prompt.
        prompt = chat_prefix
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

        # Get the completion from co:here.
        response = co.generate(model='xlarge',
                            prompt=prompt,
                            max_tokens=100,
                            temperature=0.3)
        completion = response.generations[0].text
        completion = completion.split("\nHuman:")[0]

    state = state + [(text, completion)]
    return state, state


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])
    
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
            
    txt.submit(add_text, [state, txt], [state, chatbot])

            
demo.launch()