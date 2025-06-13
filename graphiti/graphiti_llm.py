import logging
import typing
from typing import ClassVar
import logging
from collections.abc import Iterable
from typing import Union, List

import numpy as np
from pydantic import BaseModel

from graphiti_core.llm_client import LLMClient, LLMConfig
from graphiti_core.prompts.models import Message
from graphiti_core.llm_client.client import LLMClient, MULTILINGUAL_EXTRACTION_RESPONSES
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError, RefusalError
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

from pm_lida import CommonCompSettings, LlmPreset, LlmManagerLLama

llama = LlmManagerLLama()

logger = logging.getLogger(__name__)


class LlamaCppClient(LLMClient):
    """
    LlamaCppClient is a Graphiti LLMClient implementation that
    delegates to a local llama.cpp model via your llmmanager.
    """
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
    ):
        if cache:
            raise NotImplementedError("Caching is not implemented for LlamaCppClient")

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        self.manager = llama
        self.max_tokens = config.max_tokens

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        # Clean & package messages for your llama manager
        inp = [
            (m.role, self._clean_input(m.content))
            for m in messages
            if m.role in ("system", "user")
        ]

        comp_settings = CommonCompSettings(
            max_tokens=4096, # graphiti wants 16k per default
            temperature=self.temperature,
            repeat_penalty=1.0,
            stop_words=[],
        )

        if response_model is not None:
            content, calls = self.manager.completion_tool(
                preset=LlmPreset.Default,
                inp=inp,
                comp_settings=comp_settings,
                tools=[response_model],
                discard_thinks=True,
                log_file_path=None,
            )
            return calls[0].model_dump()
        else:
            content = self.manager.completion_text(
                preset=LlmPreset.Default,
                inp=inp,
                comp_settings=comp_settings,
                discard_thinks=True,
                log_file_path=None,
            )
            return {"text": content}


    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        retry_count = 0
        last_error: Exception | None = None

        messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        while retry_count <= self.MAX_RETRIES:
            try:
                return await self._generate_response(
                    messages, response_model, max_tokens, model_size
                )
            except (RateLimitError, RefusalError):
                # Bail out on these immediately
                raise
            except Exception as e:
                last_error = e
                if retry_count >= self.MAX_RETRIES:
                    logger.error(f"Max retries ({self.MAX_RETRIES}) exceeded: {e}")
                    raise

                retry_count += 1
                logger.warning(
                    f"Retrying ({retry_count}/{self.MAX_RETRIES}) after error: {e}"
                )

                # inject a little context for the next pass
                error_msg = (
                    f"The previous response attempt failed with {e.__class__.__name__}: "
                    f"{e}. Please try again."
                )
                messages.append(Message(role="user", content=error_msg))

        raise last_error or Exception("Exceeded max retries without specific error")



class LlamaCppEmbedderConfig(EmbedderConfig):
    pass

class LlamaCppEmbedder(EmbedderClient):
    """
    Graphiti EmbedderClient that delegates to a local llama.cpp model
    via your llmmanager.get_embedding().
    """
    def __init__(
        self,
        config: LlamaCppEmbedderConfig | None = None,
    ):
        if config is None:
            config = LlamaCppEmbedderConfig()
        super().__init__()

        self.manager = llama

    async def create(
        self,
        input_data: Union[
            str,
            List[str],
            Iterable[int],
            Iterable[Iterable[int]]
        ]
    ) -> List[float]:
        """
        Embed a single item. If given a list of strings, embeds all and
        returns the first embedding (to match OpenAIEmbedder.create).
        """
        # string input → single embedding
        if isinstance(input_data, str):
            emb = self.manager.get_embedding(input_data)

        # list of str → batch, but return only the first vector
        elif isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
            batch = [self.manager.get_embedding(x) for x in input_data]
            emb = batch[0]

        else:
            raise TypeError(f"Unsupported input_data for LlamaCppEmbedder.create(): {type(input_data)}")

        return emb

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """
        Embed a batch of strings, returning one vector per input.
        """
        return [
            self.manager.get_embedding(text)
            for text in input_data_list
        ]



class LlamaCppRerankerClient(CrossEncoderClient):
    """
    Graphiti CrossEncoderClient that uses embedding-based cosine similarity
    via your llmmanager.get_embedding() to score & rank passages.
    """
    def __init__(
        self,
        config: typing.Any = None,
    ):
        """
        Initialize the reranker. The config parameter is accepted
        for compatibility but not used.
        """
        # no need to call super().__init__(config) unless CrossEncoderClient requires it
        self.manager = llama

    async def rank(
        self,
        query: str,
        passages: List[str]
    ) -> List[typing.Tuple[str, float]]:
        """
        Compute embeddings for the query and each passage, then
        rank by cosine similarity score.

        Args:
            query: the input query string
            passages: list of passage strings to rank

        Returns:
            A list of (passage, score) tuples sorted descending by score.
        """
        try:
            # get query embedding
            q_emb = np.array(self.manager.get_embedding(query), dtype=float)
            q_norm = np.linalg.norm(q_emb) or 1.0

            results: List[typing.Tuple[str, float]] = []
            for passage in passages:
                p_emb = np.array(self.manager.get_embedding(passage), dtype=float)
                p_norm = np.linalg.norm(p_emb) or 1.0

                # cosine similarity
                score = float(np.dot(q_emb, p_emb) / (q_norm * p_norm))
                results.append((passage, score))

            # sort highest first
            results.sort(key=lambda pair: pair[1], reverse=True)
            return results

        except Exception as e:
            logger.error(f"Error in embedding-based reranking: {e}")
            # on failure, return passages with zero scores to preserve order
            return [(p, 0.0) for p in passages]
