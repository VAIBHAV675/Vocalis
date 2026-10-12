import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "kev"))

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from kev.checkpoint import Checkpoint
from kev.api import SystemOneRequest, to_record, to_answers
from kev.model import admit, DecisionModel, load_tokenizer

ck = Checkpoint("jaredpalmer/kev-0.8b")
print(f"[+] Loaded meta: base={ck.meta.base}, lora={ck.meta.lora}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)
raw_causal = AutoModelForCausalLM.from_pretrained(
    ck.meta.base,
    revision=ck.meta.base_revision,
    quantization_config=bnb_config,
    device_map="cuda:0"
)

# Instantiate DecisionModel
m = DecisionModel(
    ck.meta.base,
    tok,
    "cuda:0",
    lora=None,
    revision=ck.meta.base_revision,
    head_dim=ck.meta.head_dim,
    option_isolation=ck.meta.option_isolation,
    dtype=torch.float16
)
# Swap lm with 4-bit quantized base and attach LoRA
m.lm = PeftModel.from_pretrained(raw_causal.model, ck.path, torch_device="cuda:0")
m.head.load_state_dict(ck.meta.head)
m.eval()
m.head.temperature = ck.meta.temperature

print(f"[+] Kev-0.8B 4-bit INT4 (NF4) assembled! VRAM allocated: {torch.cuda.memory_allocated() / (1024*1024):.1f} MB")

# Test inference
req = SystemOneRequest(
    state="Speaker transcript: 「Hello, thank you for taking my call.」\nAcoustic speech emotion prior: neutral\nProsodic acoustic features: Normal",
    model="kev-latest",
    questions={
        "emotion": {
            "type": "choice",
            "instructions": "Emotion?",
            "criteria": {
                "neutral": "Calm routine query",
                "happy": "Delight"
            }
        }
    }
)
rec, meta = to_record(req)
enc = admit(m, tok, rec)
with torch.no_grad():
    ps = m.probs(enc)
ans = to_answers([p.tolist() for p in ps], meta)
print("[+] Test inference output:", ans)
