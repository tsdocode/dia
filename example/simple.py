from dia.model import Dia


model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

output = model.generate(text, use_torch_compile=True, verbose=True)

model.save_audio("simple.mp3", output)
