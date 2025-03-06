from pydantic import BaseModel

#model class
class TextInput(BaseModel):
    text: str
