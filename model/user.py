from pydantic import BaseModel

class User(BaseModel):
    id: int
    username: str
    email: str
    full_name: str | None = None
    disabled: bool = False
