import asyncio
import importlib


DEFAULT_MODEL = "gpt-5.4"


def _provider(name):
    try:
        return importlib.import_module(f".{name}", package=__package__)
    except (ImportError, TypeError):
        return importlib.import_module(name)

def get_answer(question, image=None, images=None, system_prompt=None, model=None):

    if model is None:
        model = DEFAULT_MODEL
    print(model)
    if model.startswith('gpt'):
        output = _provider("gpt").get_answer(question, image=image, system_prompt=system_prompt, model=model)
    elif model.startswith('claude'):
        output = _provider("claude").get_answer(question, system_prompt=system_prompt, model=model)
    elif model.startswith('deepseek'):
        output = _provider("deepseek").get_answer(question, system_prompt=system_prompt, model=model)
    elif model.startswith('qwen'):
        output = _provider("qwen").get_answer(question, image=image, images=images, system_prompt=system_prompt, model=model)
    elif model.startswith('gemini'):
        output = _provider("gemini").get_answer(question, system_prompt=system_prompt, model=model)
    else:
        output = _provider("gpt").get_answer(question, system_prompt=system_prompt, model=model)

    return output


async def async_get_answer(question, model=None, system_prompt=None):
    if model is None:
        model = DEFAULT_MODEL
    print(model)

    return await asyncio.to_thread(get_answer, question, system_prompt=system_prompt, model=model)
