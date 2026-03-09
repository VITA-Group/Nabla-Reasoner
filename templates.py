import torch

import pdb


def format_with_template(tokenizer, prompt, response=None, system_prompt=None):
    messages = [
        {"role": "user", "content": prompt},
    ]
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": system_prompt})
    if response is not None:
        messages.append({"role": "assistant", "content": response})

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=(response is None))

class Template:
    def __init__(self):
        raise NotImplemented("Base model does not implement method.")

    def apply_to_token_ids(self, *args, **kwargs):
        raise NotImplemented("Base model does not implement method.")
    
    def apply(self, *args, **kwargs):
        raise NotImplemented("Base model does not implement method.")


class GenerationTemplate(Template):
    def __init__(self, prompt, generated_token_ids, lm_model, lm_tokenizer, system_prompt=None):
        self.prompt = prompt
        if lm_model is not None:
            self.embed_in = lm_model.get_input_embeddings()
        else:
            self.embed_in = None
        self.tokenizer = lm_tokenizer

        self.prompt_text = format_with_template(lm_tokenizer, prompt, system_prompt=system_prompt)
        self.prompt_ids = lm_tokenizer.encode(self.prompt_text, add_special_tokens=False, return_tensors="pt")
        if generated_token_ids is not None:
            self.prompt_ids = torch.cat([self.prompt_ids.to(generated_token_ids.device), generated_token_ids], dim=-1)
        if self.embed_in is not None:
            self.prompt_embeds = self.embed_in(self.prompt_ids.to(self.embed_in.weight.device))
        else:
            self.prompt_embeds = None

    def apply_to_token_ids(self, token_ids):
        return torch.cat([
            self.prompt_ids.to(token_ids.device),
            token_ids,
        ], -1)

    def apply(self, soft_token):
        assert self.prompt_embeds is not None, "Language model is not specified!"
        return torch.cat([self.prompt_embeds.to(soft_token.device), soft_token], -2)


class ORMTemplate(Template):
    def __init__(self, prompt, generated_token_ids, rm_model, rm_tokenizer, system_prompt=None):
        self.prompt = prompt
        if rm_model is not None:
            self.embed_in = rm_model.get_input_embeddings()
        else:
            self.embed_in = None
        self.tokenizer = rm_tokenizer

        # parse the templates
        placeholder = "<<<$$!!PLACEHOLDER!!$$>>>"
        conv_formatted = format_with_template(rm_tokenizer, prompt, response=placeholder)
        # messages = [
        #     {"role": "user", "content": prompt},
        #     {"role": "assistant", "content": placeholder}
        # ]
        # if system_prompt is not None:
        #     messages.insert(0, {"role": "system", "content": system_prompt})
        # conv_formatted = rm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        splits = conv_formatted.split(placeholder)
        assert len(splits) == 2 or len(splits) == 1

        self.prefix_prompt_text = splits[0]
        self.postfix_prompt_text = splits[1] if len(splits) == 2 else ""

        self.prefix_prompt_ids = rm_tokenizer.encode(self.prefix_prompt_text, add_special_tokens=False, return_tensors="pt")
        if generated_token_ids is not None:
            self.prefix_prompt_ids = torch.cat([self.prefix_prompt_ids.to(generated_token_ids.device), generated_token_ids], dim=-1)
        self.postfix_prompt_ids = rm_tokenizer.encode(self.postfix_prompt_text, add_special_tokens=False, return_tensors="pt")

        if self.embed_in is not None:
            self.prefix_prompt_embeds = self.embed_in(self.prefix_prompt_ids.to(self.embed_in.weight.device))
            self.postfix_prompt_embeds = self.embed_in(self.postfix_prompt_ids.to(self.embed_in.weight.device))
        else:
            self.prefix_prompt_embeds = None
            self.postfix_prompt_embeds = None

    def apply_to_token_ids(self, token_ids):
        return torch.cat([
            self.prefix_prompt_ids.to(token_ids.device),
            token_ids,
            self.postfix_prompt_ids.to(token_ids.device),
        ], -1)

    def apply(self, soft_token):
        assert self.prefix_prompt_embeds is not None and self.postfix_prompt_embeds is not None, "Reward model is not specified!"
        return torch.cat([
            self.prefix_prompt_embeds.to(soft_token.device),
            soft_token,
            self.postfix_prompt_embeds.to(soft_token.device),
        ], -2)

