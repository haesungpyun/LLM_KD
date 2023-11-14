from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
import json

template = """You are an evaluator who rates the quality of translations from English to German.
Evaluate each translation in terms of appropriateness, content, grammar and relevance and tell each rationales. Then, please evaluate the quality of translation with a score between 0 and 10.

Here’s an evaluation example, when The English sentence is 'I always say when one door closes, another one opens.' and the translated sentence is ‘Ich sage immer, wenn eine Tür geschlossen wird, dann öffnet sich eine andere.’.

example = {{"appropriateness": 8,
"content": 9,
"grammar": 9,
"relevance": 9,
"Score": 8.5}}

Evaluate the translation of {input_text} into sentence {output_text} in JSON format:"""

prompt = PromptTemplate(template=template, input_variables=["input_text", "output_text"])
n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./model/llama-2-7b-chat.Q2_K.gguf",
    max_tokens=1024,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    verbose=False,  # Verbose is required to pass to the callback manager
    grammar_path="json.gbnf"
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

def llama(input_text, output_text):
    response = llm_chain.run({"input_text": input_text, "output_text": output_text})
    # response_sample: {"appropriateness": 7, "content": 8,"grammar": 9,"relevance": 8,"Score": 7}
    response = json.loads(response)
    return 10 - float(response["Score"])