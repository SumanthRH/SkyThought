



from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Decide on a token limit for thinking; As the model's max tokens is 32768, 32000 usually ensures there is enough space for the model to still answer
MAX_TOKENS_THINKING = 32000
SPARE_TOKENS = 300
# Decide how often to ignore end-of-thinking token
NUM_IGNORE = 1

model = LLM(
    "Qwen/Qwen2.5-Math-7B-Instruct",
    tensor_parallel_size=2,
)
tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Math-7B-Instruct"
)

stop_token_ids = tok("<|im_end|>")["input_ids"]
sampling_params = SamplingParams(
    max_tokens=32768,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
    skip_special_tokens=False,
    temperature=0.0,
)

# For the exact raspberry sample in the paper, change
# model to `qfq/1k_qr_bt_dm_po_steps` (an earlier version of s1)
# & prompt to `How many r in raspberry?`
prompts = [
    "How many r's are in raspberry",
]

system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def get_responses(conversations, llm, max_tokens, temp, args):
    if not args.budget_forcing:
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temp)
        llm.chat(
            messages=conversations, sampling_params=sampling_params, use_tqdm=True
        )
    else:
        return get_responses_budget_forcing(conversations, llm, max_tokens, temp, args)


def get_responses_budget_forcing(conversations, llm, max_tokens, temp, args): 
    # TODO: make this model agnostic
    stop_token_ids = tok.encode("<|im_start|><|im_end|>")
    think_max_tokens= max_tokens - SPARE_TOKENS
    sampling_params = SamplingParams(
        max_tokens=think_max_tokens,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=temp,
        )
    outputs = []
    for conversation in conversations:
        kwargs = {"continue_final_message": True, "add_generation_prompt": False} if conversation[-1]["role"] == "assistant" else {"add_generation_prompt": True}
        prompt = tok.apply_chat_template(conversation, tokenize=False, **kwargs)
        o = llm.generate(
            prompt,
            sampling_params=sampling_params,
        )
        ignore_str = "Wait"
        max_tokens_thinking_tmp = MAX_TOKENS_THINKING
        # Num of times to skip stop token
        for i in range(NUM_IGNORE):
            max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
            # conversation[-1]["content"] += o[0].outputs[0].text + ignore_str
            prompt += o[0].outputs[0].text + ignore_str
            sampling_params = SamplingParams(
                max_tokens=max_tokens_thinking_tmp,
                min_tokens=1,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature=0.0,
            )
            o = llm.generate(
                prompt,
                sampling_params=sampling_params,
            )
        ### Final answer ###
        # conversation[-1]["content"] += o[0].outputs[0].text
        prompt += o[0].outputs[0].text
        stop_token_ids = tok("<|im_end|>")["input_ids"]
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            min_tokens=0,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=0.0,
        )
        o = llm.generate(
            prompt,
            sampling_params=sampling_params,
        )
        # print("With budget forcing:")
        # print(prompt + o[0].outputs[0].text)
        outputs.append(prompt + o[0].outputs[0].text)
    return outputs

