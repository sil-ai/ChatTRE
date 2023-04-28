import modal


volume = modal.SharedVolume().persist("pytorch-model-vol")
CACHE_PATH = "/root/model_cache"

stub = modal.Stub("semantic-embeddings",
                     image = modal.Image.debian_slim().pip_install(
                        "pandas==1.4.3",
                        "torch==1.12.0",
                        "transformers==4.21.0",
                )
)


class SemanticSimilarity:
    def __init__(self, cache_path=CACHE_PATH):
        from transformers import BertTokenizerFast, BertModel
        try:
            self.semsim_model = BertModel.from_pretrained('setu4993/LaBSE', cache_dir=cache_path).eval()
        except OSError as e:
            print(e)
            print('Downloading model instead of using cache...')
            self.semsim_model = BertModel.from_pretrained('setu4993/LaBSE', cache_dir=cache_path, force_download=True).eval()
        print('Semantic model initialized...')

        try:
            self.semsim_tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE', cache_dir=cache_path)
        except OSError as e:
            print(e)
            print('Downloading tokenizer instead of using cache...')
            self.semsim_tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE', cache_dir=cache_path, force_download=True)
        print('Tokenizer initialized...')

    @stub.function(
            timeout=3600,
            )
    def predict(self, sent1: str, sent2: str, precision: int=2):
        import torch
        """
        Return a prediction.

        Parameters
        ----------
        sent1, sent2 : 2 lists of verse strings to be compared
        
        returns sentences plus a score
        """
        try:
            sent1_input = self.semsim_tokenizer(sent1, return_tensors="pt", padding=False, truncation=True)
        except:
            print('error tokenizing sent1')
            print(sent1)
            return None
        with torch.no_grad():
            sent1_output = self.semsim_model(**sent1_input)

        sent1_embedding = sent1_output.pooler_output

        return sent1_embedding


@stub.function(
        timeout=3600, 
        shared_volumes={CACHE_PATH: volume},
)
def get_embeddings(text_list):
   
    sem_sim = SemanticSimilarity(cache_path=CACHE_PATH)

    #default offset is all of the verses
    sents1 = text_list
    sents2 = text_list
    results = list(sem_sim.predict.map(sents1,sents2))
    print(results[:20])

    return results
