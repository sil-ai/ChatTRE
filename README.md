# ChatTRE
Chat with translator resources

## Usage
Run `python db.py` to create the database embeddings.

Start the API with `uvicorn api:app`
First you need to authenticate the Google Translate API, which isn't straightforward. Or just comment out the translation step and use it only in English.

Start the demo UI with `python demo_ui.py`

