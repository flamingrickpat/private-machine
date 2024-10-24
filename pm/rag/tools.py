from typing import TypedDict, Dict, List
import logging

from pydantic import BaseModel, Field

from pm.database.db_helper import search_documentation

logger = logging.getLogger(__name__)

class RagState(TypedDict):
    finished: bool
    guid_id_map: Dict[str, int]
    status: bool
    docs_temp: List[int]
    docs_final: List[int]
    current_id: int

class hybrid_search_documentation(BaseModel):
    """Perform a hybrid search on the documentation and return documents in next message."""
    query_string: str = Field(description="Can be complex query or keywords")

    def execute(self, rag_state: RagState) -> str:
        logger.info(f"Calling hybrid_search_documentation with {self.query_string}")
        items = []
        rag_state["docs_temp"] = []
        rag_data = search_documentation(self.query_string, max_size=1024)
        for doc in rag_data:
            if doc.id in rag_state["guid_id_map"].keys():
                cur_id = rag_state["guid_id_map"]
            else:
                rag_state["current_id"] += 1
                cur_id = rag_state["current_id"]
                rag_state["guid_id_map"][doc.id] = cur_id

            item = f"### BEGIN DOC\nDocumentID: {cur_id}\n{doc.text}\n### END DOC"
            items.append(item)
            rag_state["docs_temp"].append(cur_id)

        representation = "\n ".join(items)
        return representation


class add_document(BaseModel):
    """Add a document to the document list for the final answer. Status will be returned on next message."""
    document_ids: List[int] = Field(description="list of document id from the output of hybrid_search_documentation function")

    def execute(self, rag_state: RagState) -> str:
        logger.info(f"Calling add_document with {repr(self.document_ids)}")
        for doc_id in self.document_ids:
            rag_state["docs_final"].append(doc_id)
        lst = ", ".join([str(x) for x in rag_state["docs_final"]])
        return f"Successfully added, knowledge list now contains: [{lst}]"


class answer_user(BaseModel):
    """Finish the hybrid search loop and make final answer to user. This ends the current chat."""
    status: bool = Field(description="true for enough data for answer or false for impossible to answer")

    def execute(self, rag_state: RagState):
        logger.info(f"Calling answer_user with {self.status}")
        rag_state["finished"] = True
        rag_state["status"] = self.status
