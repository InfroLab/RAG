from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import os

def convert_hf_to_onnx(model_or_path, save_directory: None) -> str:
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        model_id=model_or_path,
        export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_or_path)
    if not save_directory:
        save_directory = os.path.split(model_or_path)[-1]+'-onnx'
    ort_model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print('Converted model saved')
    return save_directory

convert_hf_to_onnx('sentence-transformers/msmarco-MiniLM-L12-cos-v5')