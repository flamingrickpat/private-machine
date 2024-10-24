from pm.llm.llamacpp import LlamaCppLlm


def chat_complete(llm, call):
    if llm.__class__ == LlamaCppLlm:
        llm_lc = llm.get_langchain_model()
        ai_msg = llm_lc.invoke(call)
        return ai_msg