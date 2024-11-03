from datetime import datetime

from numpy.ma.core import where
from pydantic import Field
from lancedb.pydantic import LanceModel

from pm.controller import controller
from pm.database.db_model import get_guid, Transaction, Message, Fact, MessageSummary, ConceptualCluster, Relation, Conversation, User


# Start a new transaction by recording its start time in the Transaction table
def start_transaction():
    # Check if there's already an active transaction
    transaction_table = controller.db.open_table(Transaction.table)
    active_transaction = transaction_table.search().where("state='active'").to_pydantic(model=Transaction)
    active_transaction = active_transaction[0].dict() if len(active_transaction) > 0 else None
    if active_transaction:
        raise ValueError("An active transaction already exists. Commit or rollback before starting a new one.")

    # Record the new transaction start time
    new_transaction = Transaction(state="active")
    transaction_table.add([new_transaction])


# Commit the transaction by marking it as completed
def commit_transaction():
    transaction_table = controller.db.open_table(Transaction.table)

    # Find the active transaction
    active_transaction = transaction_table.search().where("state='active'").to_pydantic(model=Transaction)
    active_transaction = active_transaction[0] if len(active_transaction) > 0 else None
    if active_transaction:
        # Mark the transaction as completed
        transaction_table.update(where=f"id='{active_transaction.id}'", values={"state": "completed"})


# Rollback the transaction by deleting rows created during the active transaction
def rollback_transaction():
    transaction_table = controller.db.open_table(Transaction.table)

    # Find the active transaction
    active_transaction = transaction_table.search().where("state='active'").to_pydantic(model=Transaction)
    active_transaction = active_transaction[0].dict() if len(active_transaction) > 0 else None
    if not active_transaction:
        pass
    else:
        # Get the start time of the active transaction
        transaction_start_time = active_transaction["created_at"]

        # Delete rows in all relevant tables created after the transaction start time
        for table in [Message.table, Fact.table, MessageSummary.table, ConceptualCluster.table, Relation.table, Conversation.table, User.table]:
            db_table = controller.db.open_table(table)
            db_table.delete(f"created_at >= cast('{transaction_start_time.isoformat()}' as timestamp)")

        # Mark the transaction as completed
        transaction_table.update(where=f"id='{active_transaction['id']}'", values={"state": "completed"}, )
