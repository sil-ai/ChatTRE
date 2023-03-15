import os
import shutil
import uuid

import chromadb
from chromadb.config import Settings
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk

# NLTK setup
nltk.download('punkt')

# delete the .chromadb directory if you want to start fresh
if os.path.exists(".chromadb"):
    shutil.rmtree(".chromadb")

# Chroma client
os.mkdir(".chromadb")
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb" 
))

# Create a collection
collections = client.list_collections()
col_names = [col.name for col in collections]
if "tyndale" not in col_names:
    collection = client.create_collection(name="tyndale")

# Load the tyndale data
book_intro_summaries = pd.read_csv("tyndale_data/CSV/book_intro_summaries.csv")
book_intros = pd.read_csv("tyndale_data/CSV/book_intros.csv")
profiles = pd.read_csv("tyndale_data/CSV/profiles.csv")
study_notes = pd.read_csv("tyndale_data/CSV/study_notes.csv")
theme_notes = pd.read_csv("tyndale_data/CSV/theme_notes.csv")

# Process book intro summaries
docs = []
metadata = []
ids = []
for _, row in book_intro_summaries.iterrows():
  purpose = "The Purpose of " + row["Book"] + " is: " + row["Purpose"]
  author = "The Author of " + row["Book"] + " is: " + row["Author"]
  date = "The Dates of " + row["Book"] + " is: " + row["Date"]
  setting = "The Setting of " + row["Book"] + " is: " + row["Setting"]
  docs = docs + [
      purpose,
      author,
      date,
      setting
  ]
  book = row['Book'].split('The Book of ')[-1]
  metadata = metadata + [
      {
          "book": book,
          "citation": "Tyndale Open Study Notes, Book Intro Summaries, " + book
      },
      {
          "book": book,
          "citation": "Tyndale Open Study Notes, Book Intro Summaries, " + book
      },
      {
          "book": book,
          "citation": "Tyndale Open Study Notes, Book Intro Summaries, " + book
      },
      {
          "book": book,
          "citation": "Tyndale Open Study Notes, Book Intro Summaries, " + book
      },
  ]
  ids = ids + [
      str(uuid.uuid4()),
      str(uuid.uuid4()),
      str(uuid.uuid4()),
      str(uuid.uuid4())
  ]

collection.add(
    documents=docs,
    metadatas=metadata,
    ids=ids
)

# Book intros
docs = []
metadata = []
ids = []
for _, row in book_intros.iterrows():
  overview = row["Overview"]
  notes = []
  entry = ""
  for sent in sent_tokenize(row['Notes']):
    if len(entry) < 500:
      entry = entry + " " + sent
    else:
      notes.append(entry)
      entry = sent
  notes.append(entry)
  docs = docs + [overview] + notes
  book = row['Book']
  metadata = metadata + [
      {
          "book": book,
          "start_chapter": row["Start_Chapter"],
          "end_chapter": row["End_Chapter"],
          "start_verse": row["Start_Verse"],
          "end_verse": row["End_Verse"],
          "citation": "Tyndale Open Study Notes, Book Intros, " + book
      }
  ]
  for note in notes:
    metadata.append({
          "book": book,
          "citation": "Tyndale Open Study Notes, Book Intros, " + book
      })
  ids = ids + [str(uuid.uuid4())]
  for note in notes:
    ids.append(str(uuid.uuid4()))

collection.add(
    documents=docs,
    metadatas=metadata,
    ids=ids
)

# Profiles
docs = []
metadata = []
ids = []
for _, row in profiles.iterrows():
  notes = []
  entry = ""
  for sent in sent_tokenize(row['Notes']):
    if len(entry) < 500:
      entry = entry + " " + sent
    else:
      notes.append(entry)
      entry = sent
  notes.append(entry)
  docs = docs + notes
  person = row['Person']
  for note in notes:
    metadata.append({
            "book": row['Book'],
            "start_chapter": row["Start_Chapter"],
            "end_chapter": row["End_Chapter"],
            "start_verse": row["Start_Verse"],
            "end_verse": row["End_Verse"],
            "citation": "Tyndale Open Study Notes, Profiles, " + person
        })
  for note in notes:
    ids.append(str(uuid.uuid4()))

collection.add(
    documents=docs,
    metadatas=metadata,
    ids=ids
)

# Study notes
docs = []
metadata = []
ids = []
for _, row in study_notes.iterrows():
  docs.append(row['Text'])
  metadata.append({
          "book": row['Book'],
          "start_chapter": row["Start_Chapter"],
          "end_chapter": row["End_Chapter"],
          "start_verse": row["Start_Verse"],
          "end_verse": row["End_Verse"],
          "citation": "Tyndale Open Study Notes, Study Notes, " + row['vref']
      })
  ids.append(str(uuid.uuid4()))

collection.add(
    documents=docs,
    metadatas=metadata,
    ids=ids
)

# Theme notes
docs = []
metadata = []
ids = []
for _, row in theme_notes.iterrows():
  notes = []
  entry = ""
  for sent in sent_tokenize(row['Text']):
    if len(entry) < 500:
      entry = entry + " " + sent
    else:
      notes.append(entry)
      entry = sent
  notes.append(entry)
  docs = docs + notes
  title = row['Title']
  for note in notes:
    metadata.append({
            "title": title,
            "citation": "Tyndale Open Study Notes, Theme Notes, " + title
        })
  for note in notes:
    ids.append(str(uuid.uuid4()))

collection.add(
    documents=docs,
    metadatas=metadata,
    ids=ids
)

results = collection.query(
    query_texts=["Which four sources are proposed as the four sources of the Pentateuch?"],
    n_results=2
)

print(results)