from transformers import AutoModelForCausalLM, AutoTokenizer
    
model = AutoModelForCausalLM.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/mistral-7b", torch_dtype='auto')

tokenizer = AutoTokenizer.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/mistral-7b")

print(model)