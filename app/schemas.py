from pydantic import BaseModel

class OrgSignup(BaseModel):
    name: str
    email: str
    password: str

class OrgLogin(BaseModel):
    email: str
    password: str

class EmployeeSignup(BaseModel):
    org_id: int
    name: str
    email: str
    employee_id: str

class EmployeeLogin(BaseModel):
    org_id: int
    employee_id: str
