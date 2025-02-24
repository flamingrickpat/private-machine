from sqlalchemy import create_engine, text
from sqlmodel import SQLModel, Session

from pm.character import database_uri
from pm.controller import controller


def update_database_item(event: SQLModel):
    if event is None:
        return

    session = controller.get_session()
    session.add(event)
    session.flush()
    session.refresh(event)
    return event.id


def get_engine():
    return create_engine(database_uri)

def init_db():
    engine = get_engine()
    try:
        with Session(engine) as session:
            session.exec(text("CREATE EXTENSION vector;"))
            session.commit()
    except:
        pass

    reset = False

    try:
        with Session(engine) as session:
            if reset:
                session.exec(text("drop table event;"))
                session.exec(text("drop table cluster;"))
                session.exec(text("drop table eventcluster;"))
                session.exec(text("drop table fact;"))
                session.commit()

        SQLModel.metadata.create_all(engine)
    except:
        pass
