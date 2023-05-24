from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to(device)


input_text = "The quick brown fox jumps."
num_iterations = 5000
inference_timings = []

for _ in range(num_iterations):
    # Tokenize input text and convert to PyTorch tensors
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    start_time = time.perf_counter()
    outputs = model(**inputs)
    end_time = time.perf_counter()

    duration = (end_time - start_time) * 1000  # Convert to milliseconds
    inference_timings.append(duration)

    del inputs, outputs
    torch.cuda.empty_cache()

with open("inference_timings.txt", "w") as file:
    for timing in inference_timings:
        file.write(str(timing) + "\n")
