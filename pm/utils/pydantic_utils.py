from pydantic import BaseModel, create_model, Field
import random
import string
from copy import deepcopy


def create_basemodel(field_definitions, randomnize_entries=False):
    """
    Create a dynamic Pydantic BaseModel with randomized field order.

    Args:
        field_definitions (list): A list of dictionaries where each dict should have:
            - 'name': the field name (str)
            - 'type': the field type (e.g. str, int, etc.)
            - 'description': a description for the field (str)
        randomnize_entries (bool): If True, each field gets a random default value
            based on its type. Otherwise, the field is required.

    Returns:
        A dynamically created Pydantic BaseModel.
    """
    # Make a deep copy and randomize the order
    randomized_fields = deepcopy(field_definitions)
    if randomnize_entries:
        random.shuffle(randomized_fields)

    fields = {}
    for field in randomized_fields:
        name = field['name']
        f_type = field['type']
        description = field.get('description', "")

        # Use Ellipsis to indicate the field is required.
        fields[name] = (f_type, Field(..., description=description))

    # Create and return a dynamic model class
    return create_model('DynamicBaseModel', **fields)
