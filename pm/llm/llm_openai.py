import logging

from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_delay, wait_fixed

logger = logging.getLogger(__name__)

def get_llm(model, comp_settings):
    llm = ChatOpenAI(
        model=model.model,
        temperature=comp_settings.temperature if comp_settings.temperature is not None else 0.6,
        max_tokens=comp_settings.max_tokens if comp_settings.max_tokens is not None else 1024,
        timeout=None,
        max_retries=2,
        api_key=model.api_key,
        base_url=model.base_url
    )
    return llm

@retry(stop=stop_after_delay(2000), wait=wait_fixed(60))
def chat_complete_openai(llm, messages: list[tuple[str, str]]):
    logger.info("\n".join([f"{x[0]}: {x[1]}" for x in messages]))
    ai_msg = llm.invoke(messages)
    logger.info(ai_msg.content)
    return ai_msg
