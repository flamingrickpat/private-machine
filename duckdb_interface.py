import duckdb
from pm.database.db_model import User, Message, Conversation, ConceptualCluster, MessageSummary, Relation, Fact, Transaction
from pm.controller import controller

controller.start()

user = controller.db.open_table(User.table).to_lance()
message = controller.db.open_table(Message.table).to_lance()
conversation = controller.db.open_table(Conversation.table).to_lance()
conceptual_cluster = controller.db.open_table(ConceptualCluster.table).to_lance()
message_summary = controller.db.open_table(MessageSummary.table).to_lance()
relation = controller.db.open_table(Relation.table).to_lance()
fact = controller.db.open_table(Fact.table).to_lance()
transaction = controller.db.open_table(Transaction.table).to_lance()

print("ready...")
while True:
    try:
        query = input()
        res = duckdb.query(query)
        print(len(res))
        print(res)
    except Exception as e:
        print(e)
