import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# Укажи нужную модель (например, Llama-2, Falcon, или любую инструкционную)
model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

textgen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=64,
    pad_token_id=tokenizer.eos_token_id
)


# Загрузи abstracts.json (твой файл с аннотациями)
with open("./Im_professional/12/data/abstracts.json", "r", encoding="utf-8") as f:
    abstracts = json.load(f)

submit = {}

for idx, text in tqdm(abstracts.items()):
    prompt = (
        "Analyze the following scientific abstract and decide if it was written by an AI or a human. "
        "Consider typical AI markers such as unnatural repetition, excessive use of asterisks (**), overly formal language, generic structure, or formulaic phrasing. "
        "If any of these features are present and suggest AI authorship, answer only 'true'. If the text resembles authentic human scholarly writing, answer only 'false'.\n\n"
        f"{text}\n\nAnswer (true for AI, false for human):"
    )

    output = textgen(prompt)[0]['generated_text']
    # Преобразуем ответ к формату
    response = output.strip().lower()
    if 'true' in response:
        submit[idx] = True
    elif 'false' in response:
        submit[idx] = False
    else:
        # fallback: если модель "заболтала", лучше вручную обработать
        submit[idx] = False

# Сохрани результат в submit.json
with open("submit.json", "w", encoding="utf-8") as f:
    json.dump(submit, f, indent=2)

print("Готово! Сгенерирован submit.json в нужном формате.")
