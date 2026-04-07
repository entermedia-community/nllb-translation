from typing import Union, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacremoses import MosesPunctNormalizer
from stopes.pipelines.monolingual.utils.sentence_split import get_split_algo
import torch
import nltk
from functools import lru_cache

nltk.download("punkt_tab")

app = FastAPI(title="eMedia Translation API")

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "facebook/nllb-200-3.3B"

def load_model():
  model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
  print(f"Model loaded in {device}")
  return model

model = load_model()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

punct_normalizer = MosesPunctNormalizer(lang="en")


@lru_cache(maxsize=202)
def get_sentence_splitter(language_code: str):
  short_code = language_code[:3]
  return get_split_algo(short_code, "default")


available_languages = {
  "en": "eng_Latn",
  "fr": "fra_Latn",
  "de": "deu_Latn",
  "es": "spa_Latn",
  "pt": "por_Latn",
  "pt_br": "por_Latn",
  "ru": "rus_Cyrl",
  "zh": "zho_Hans",
  "zht": "zho_Hant",
  "hi": "hin_Deva",
  "ar": "arb_Arab",
  "bn": "ben_Beng",
  "ur": "urd_Arab",
  "sw": "swh_Latn"
}

@lru_cache(maxsize=64)
def translate_text(text: str, src: str, target: str, max_length: Optional[int] = None) -> str:
  tokenizer.src_lang = src

  text = punct_normalizer.normalize(text)

  target_id = tokenizer.convert_tokens_to_ids(target)
  paragraphs = text.split("\n")
  translated_paragraphs = []

  for paragraph in paragraphs:
    splitter = get_sentence_splitter(src)
    sentences = list(splitter(paragraph))
    translated_sentences = []

    for sentence in sentences:
      input_ids = (
        tokenizer(sentence, return_tensors="pt")
        .input_ids[0]
        .cpu()
        .numpy()
        .tolist()
      )
      chunk_max_length = max_length if max_length is not None else len(input_ids) + 50
      generated_tokens = model.generate(
        input_ids=torch.tensor([input_ids]).to(device),
        forced_bos_token_id=target_id,
        max_length=chunk_max_length,
        num_return_sequences=1,
        num_beams=5,
        no_repeat_ngram_size=4,
        renormalize_logits=True,
      )
      translated_sentences.append(
        tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
      )

    translated_paragraphs.append(" ".join(translated_sentences))

  return "\n".join(translated_paragraphs)

@app.get("/")
def read_root():
  return {"message": "Welcome to the eMedia Translation API"}

@app.get("/health")
@app.get("/health.ico")
def health_check():
  return {"status": "ok", "model": MODEL_NAME, "device": device}


def verify_langs(source: str, targets: List[str]) -> Union[bool, str]:
  if source not in available_languages.keys():
    return False, f"Source language '{source}' is not supported. Available languages: {', '.join(available_languages)}"
  
  for target in targets:
    if target not in available_languages.keys():
      return False, f"Target language '{target}' is not supported. Available languages: {', '.join(available_languages)}"
  
  return True, ""

class TranslateRequest(BaseModel):
  q: Union[str, List[str]] = Field(..., description="Text or list of texts to translate")
  source: str = Field(..., description="Source language code, e.g. 'en'")
  target: Union[str, List[str]] = Field(..., description="Target language code(s), e.g. 'fr' or ['fr', 'de']")
  max_length: Optional[int] = Field(None, description="Maximum length of the translated text")

class TranslationResponse(BaseModel):
  translatedText: dict = Field(..., description="Dictionary with target language codes as keys and list of translated texts as values")

@app.post("/translate")
def translate(req: TranslateRequest):
  text_arr = req.q
  if isinstance(text_arr, str):
    text_arr = [text_arr]
  
  source = req.source.lower()
  targets_arr = req.target

  if isinstance(targets_arr, str):
    targets_arr = [targets_arr.lower()]
  
  valid, err = verify_langs(source, targets_arr)
  if not valid:
    return {"error": err}

  try:
    result = {}
    for target_current in targets_arr:

      target = target_current.lower()

      result[target_current] = []

      for text in text_arr:
        translation = translate_text(text, src=available_languages[source], target=available_languages[target], max_length=req.max_length)
        result[target_current].append(translation)

    return TranslationResponse(translatedText=result)

  except Exception as e:
    raise HTTPException(
      status_code=500, 
      detail=f"Translation error: {str(e)}"
    )
