from aitextgen import aitextgen

ai= aitextgen()

prompt = "cricket is"

gpt_text = ai.generate_one(prompt = prompt , max_length=100)

print(gpt_text)
