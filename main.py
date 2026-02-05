from typing import Union, List, Optional
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = os.path.expanduser("~/models")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from functools import lru_cache

app = FastAPI(title="eMedia Translation API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "facebook/nllb-200-1.3B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", dtype=torch.float16)
model.to(DEVICE)

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
  if max_length is None:
    max_length = len(text.split()) * 2 + 50
  tokenizer.src_lang = src
  inputs = tokenizer(
    text, 
    return_tensors="pt", 
    padding=True, 
    truncation=True, 
    max_length=max_length
  ).to(DEVICE)

  target_id = tokenizer.convert_tokens_to_ids(target)
  generated_tokens = model.generate(
    **inputs, 
    forced_bos_token_id=target_id, 
    num_beams=5, 
    early_stopping=True, 
    max_length=max_length
  )
  return tokenizer.batch_decode(
    generated_tokens, 
    skip_special_tokens=True
  )[0]

@app.get("/")
def read_root():
  return {"message": "Welcome to the eMedia Translation API"}

@app.get("/health")
@app.get("/health.ico")
def health_check():
  return {"status": "ok", "model": model_name, "device": DEVICE}


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
