# main.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
import os
import getpass
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv



load_dotenv()


# -------------------------------------------------
# 🔹 API & Model Initialization
# -------------------------------------------------
app = FastAPI(title="Pakistani Legal AI API",
              description="Classifies and prioritizes Pakistani legal cases using LangChain & Gemini",
              version="1.0")

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter Google Gemini API Key: ")

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


# -------------------------------------------------
# 🔹 1. Legal Classification Chain
# -------------------------------------------------
classification_examples = [
    {"case": "A person arrested under Section 302 PPC for murder. Bail application filed in Sessions Court.",
     "classification": "Criminal Law"},
    {"case": "میاں بیوی میں طلاق کا معاملہ۔ بیوی حق مہر اور کسٹڈی کا مطالبہ کر رہی ہے۔",
     "classification": "Family Law"},
    {"case": "FBR issued tax demand of Rs. 30 crore. Businessman filed appeal claiming excessive assessment.",
     "classification": "Tax Law"},
    {"case": "Writ petition under Article 199 challenging illegal detention by police. Fundamental rights violated.",
     "classification": "Constitutional Law"},
    {"case": "Government officer dismissed from service without inquiry. Filed appeal in Service Tribunal.",
     "classification": "Service Law"},
    {"case": "Landlord filed eviction suit against tenant for 10 months rent default. Property in Karachi.",
     "classification": "Property Law"},
    {"case": "50 factory workers terminated illegally. Labour Court ordered reinstatement. Employer appealed.",
     "classification": "Labour Law"},
    {"case": "Company director sued for breach of fiduciary duty and misappropriation of Rs. 50 million.",
     "classification": "Corporate Law"}
]

classification_example_template = """
Case: {case}
Classification: {classification}
"""

classification_prompt = FewShotPromptTemplate(
    examples=classification_examples,
    example_prompt=PromptTemplate(
        input_variables=["case", "classification"],
        template=classification_example_template
    ),
    prefix="""You are an expert Pakistani legal case classifier with deep knowledge of Pakistan's legal system.
Classify the case into one of the following categories:
1. Criminal Law
2. Civil Law
3. Family Law
4. Constitutional Law
5. Corporate Law
6. Labour Law
7. Property Law
8. Tax Law
9. Banking Law
10. Service Law
11. Islamic Law
12. Administrative Law
13. Environmental Law
14. Intellectual Property
15. Election Law

Respond in this EXACT format:
Category: [category name]
Reasoning: [2-3 sentences explaining why this category fits]

EXAMPLES:""",
    suffix="""
NOW CLASSIFY THIS CASE:
{input}

Category: 
Reasoning:
""",
    input_variables=["input"]
)

classification_chain = LLMChain(llm=model, prompt=classification_prompt)


# -------------------------------------------------
# 🔹 2. Priority Assessment Chain
# -------------------------------------------------
priority_examples = [
    {"case": "Murder accused arrested under Section 302 PPC. Bail hearing scheduled in 3 days. Accused in jail for 2 months.",
     "priority": "High"},
    {"case": "Property dispute over boundary wall between neighbors. Case pending for 5 years. No immediate harm.",
     "priority": "Low"},
    {"case": "Habeas corpus petition filed. Person detained illegally by police for 10 days without FIR. Family has no contact.",
     "priority": "Critical"},
    {"case": "Wife seeking maintenance for herself and 3 minor children. Husband not paying for 8 months. Children's education affected.",
     "priority": "High"},
    {"case": "Tax appeal against FBR assessment. Amount Rs. 5 lakh. Appeal deadline in 45 days.",
     "priority": "Medium"},
]

priority_example_template = """
Case: {case}
Priority: {priority}
"""

priority_prompt = FewShotPromptTemplate(
    examples=priority_examples,
    example_prompt=PromptTemplate(
        input_variables=["case", "priority"],
        template=priority_example_template
    ),
    prefix="""You are an expert Pakistani legal case priority assessor.
Decide the urgency level: CRITICAL, HIGH, MEDIUM, or LOW.

Factors:
- Life or liberty at risk → Critical
- Fundamental rights violations → High or Critical
- Deadlines within days/weeks → High
- Routine matters → Low

Respond in this EXACT format:
Priority Level: [CRITICAL/HIGH/MEDIUM/LOW]
Key Factors: [List of 3–4 points]
Recommended Action Timeline: [e.g., Within 3 days, Within 2 weeks, etc.]
Reasoning: [2–3 sentences]
""",
    suffix="""
NOW ASSESS THIS CASE:
{input}

Priority Level: 
Key Factors: 
Recommended Action Timeline: 
Reasoning:
""",
    input_variables=["input"]
)

priority_chain = LLMChain(llm=model, prompt=priority_prompt)


# -------------------------------------------------
# 🔹 Request Schemas
# -------------------------------------------------
class CaseInput(BaseModel):
    text: str


# -------------------------------------------------
# 🔹 API Endpoints
# -------------------------------------------------
@app.post("/classify")
def classify_case(case: CaseInput):
    """Classify a Pakistani legal case."""
    result = classification_chain.run(input=case.text)
    return {"classification_result": result}


@app.post("/priority")
def assess_priority(case: CaseInput):
    """Assess the urgency and priority level of a case."""
    result = priority_chain.run(input=case.text)
    return {"priority_result": result}


@app.post("/analyze")
def full_analysis(case: CaseInput):
    """Get both classification and priority in one response."""
    classification = classification_chain.run(input=case.text)
    priority = priority_chain.run(input=case.text)
    return {
        "classification": classification,
        "priority": priority
    }


# -------------------------------------------------
# 🔹 Root
# -------------------------------------------------
@app.get("/")
def root():
    return {"message": "Pakistani Legal AI API is running. Endpoints: /classify, /priority, /analyze"}
