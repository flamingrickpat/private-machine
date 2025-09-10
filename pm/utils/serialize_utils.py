import logging
import logging
import pickle
from typing import List
from typing import (
    Optional,
)


# Helper function to serialize embeddings (adapted from your example)
def serialize_embedding(value: Optional[List[float]]) -> Optional[bytes]:
    if value is None or not value:  # Handle empty list too
        return None
    # Using pickle as requested, but consider JSON for cross-language/version compatibility if needed
    return pickle.dumps(value)


# Helper function to deserialize embeddings
def deserialize_embedding(value: Optional[bytes]) -> List[float]:
    if value is None:
        return []  # Return empty list instead of None for consistency with model field type hint
    try:
        # Ensure we are dealing with bytes
        if not isinstance(value, bytes):
            logging.warning(f"Attempting to deserialize non-bytes value as embedding: {type(value)}. Trying to encode.")
            try:
                value = str(value).encode('utf-8')  # Attempt basic encoding
            except Exception:
                logging.error("Failed to encode value to bytes for deserialization. Returning empty list.")
                return []
        return pickle.loads(value)
    except pickle.UnpicklingError as e:
        logging.error(f"Failed to deserialize embedding (UnpicklingError): {e}. Value prefix: {value[:50] if value else 'None'}. Returning empty list.")
        return []
    except Exception as e:  # Catch other potential errors like incorrect format
        logging.error(f"Failed to deserialize embedding (General Error): {e}. Value prefix: {value[:50] if value else 'None'}. Returning empty list.")
        return []

