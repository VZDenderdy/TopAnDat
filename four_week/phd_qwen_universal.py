import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from GPTID.IntrinsicDim import PHD

# === Настройки PHD ===
MIN_SUBSAMPLE = 40
INTERMEDIATE_POINTS = 7

# === Загрузка модели ===
def load_qwen_model(model_path: str, device: str = "cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    print("Модель загружена на:", model.device)
    return tokenizer, model

# === Препроцессинг ===
def preprocess_text(text):
    return text.replace('\n', ' ').replace('  ', ' ')

# === Токенизация и эмбеддинги ===
def get_embeds(text, tokenizer, model, max_length=2048, returns_tokenized=False):
    device = model.device
    inputs = tokenizer(preprocess_text(text), truncation=True,
                       max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outp = model(**inputs)

    if not returns_tokenized:
        return outp[0][0].cpu().numpy()
    
    tokens = [tokenizer.decode([tok]) for tok in inputs['input_ids'].reshape(-1)]
    return outp[0][0].cpu().numpy(), tokens

# === Один запуск PHD ===
def get_phd_single(text, solver, tokenizer, model, max_length=2048):
    device = model.device
    inputs = tokenizer(preprocess_text(text), truncation=True,
                       max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outp = model(**inputs)

    mx_points = inputs['input_ids'].shape[1]
    mn_points = MIN_SUBSAMPLE
    step = (mx_points - mn_points) // INTERMEDIATE_POINTS

    return solver.fit_transform(
        outp[0][0].cpu().numpy(),
        min_points=mn_points,
        max_points=mx_points - step,
        point_jump=step
    )

# === Усреднение по многим прогонам ===
def get_phd_single_loop(text, solver, tokenizer, model, n_tries=10, max_length=2048):
    return np.mean([
        get_phd_single(text, solver, tokenizer, model, max_length)
        for _ in range(n_tries)
    ])

# === Запуск по DataFrame ===
def get_phd(df, tokenizer, model, key='text', is_list=False, alpha=1.0, n_tries=10, max_length=2048):
    dims = []
    PHD_solver = PHD(alpha=alpha, metric='euclidean', n_points=9)
    for s in tqdm(df[key]):
        text = s[0] if is_list else s
        dims.append(get_phd_single_loop(text, PHD_solver, tokenizer, model, n_tries=n_tries, max_length=max_length))
    return np.array(dims).reshape(-1, 1)

# === Для заранее готовых эмбеддингов ===
def get_raw_phd(points, alpha=1.0):
    points = points.T
    mx_points = points.shape[1]
    mn_points = MIN_SUBSAMPLE
    step = (mx_points - mn_points) // INTERMEDIATE_POINTS
    solver = PHD(alpha=alpha, metric='euclidean', n_points=9)
    return solver.fit_transform(points.T, min_points=mn_points, max_points=mx_points - step, point_jump=step)

def get_raw_phd_in_loop(points, alpha=1.0, n_tries=10):
    solver = PHD(alpha=alpha, metric='euclidean', n_points=9)
    return [get_raw_phd(points, alpha=alpha) for _ in range(n_tries)]
    