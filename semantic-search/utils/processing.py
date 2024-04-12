import os

from langdetect import detect
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
tqdm.pandas()

#Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask) -> torch.FloatTensor:
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#Encode text
def encode(texts: list[str], model, tokenizer) -> torch.FloatTensor:
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)
    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def get_scores(query_emb: torch.FloatTensor, doc_embs: torch.FloatTensor) -> torch.FloatTensor:
    return torch.mm(query_emb, doc_embs.transpose(0, 1))[0].cpu().tolist()

def convert_cnbc_to_fulltext(path_to_csv: str, sep: str=',', quotechar: str='"') -> str:
    df = pd.read_csv(path_to_csv, sep=sep, quotechar=quotechar)
    df = df.reset_index().rename(columns={'index':'article_id'})
    df['full_text'] = df['title']+'. '+df['short_description']+'. '+df['keywords']+'. '+df['description']
    df['full_text'] = df['full_text'].str.replace('\n',' ')
    df = df[df['full_text'].notna() & (df['full_text']!='')]
    df[['article_id', 'full_text']].to_csv('cnbc_fulltext.txt', index=False)

def convert_world_news_to_fulltext(path_to_csv: str, sep: str=',', quotechar: str='"') -> str:
    df = pd.read_csv(r'C:\Users\Infro\Documents\RAG\news_api_dataset\data.csv')
    def detect_lang(text):
        try:
            return detect(text)
        except:
            return 'n/a'
    df = df[df.title.notna()]
    df['lang'] = df['title'].progress_apply(detect_lang)
    df = df[df['lang']=='en']
    def unify_content(content, full_content):
        if content in full_content:
            return full_content
        return content+'. '+full_content
    text_rows = ['content','full_content', 'title', 'description']
    df.loc[:, text_rows] = df.loc[:, text_rows].fillna('').astype(str)
    df['unified_content'] = df[['content','full_content']].progress_apply(lambda x: unify_content(x['content'], x['full_content']), axis=1)
    df['full_text'] = df['title'] + '. ' + df['description'] + '. ' + df['unified_content']
    df[['article_id','full_text']].to_csv('newsapi_100k_fulltext.txt',header=None, index=False)

def generate_article_files(df_ft: pd.DataFrame, dest_path: str) -> None:
    def save_files(article_id, full_text):
        if not isinstance(full_text, str):
            return
        with open(os.path.join(dest_path, str(article_id)), 'w', encoding='utf8') as f:
            f.write(full_text)
    _ = df_ft.progress_apply(lambda row: save_files(row['article_id'], row['full_text']), axis=1)
    print('Saved aricles:', len(os.listdir(dest_path)))