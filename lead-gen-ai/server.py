from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import openai
import os
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import autogen
import traceback
import json
import logging

# --- Setup Logging ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

#####################################################################################################
# API & GOOGLE SHEETS CONFIGURATION
#####################################################################################################
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Sheets Setup
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SERVICE_ACCOUNT_FILE = "google_sheets_credentials.json"
SHEET_ID = "1JQ13k3KJ1baInqtavH58bLwUAMcR2B0e2FrFVrD-7Fo"

creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

##################################################################################################
# LEAD DATA MODEL (Pydantic)
##################################################################################################

class Lead(BaseModel):
    name: str
    income: str
    savings: str
    credit_score: Optional[str] = None
    dob: Optional[str] = None
    lump_sum: Optional[str] = None
    monthly_contribution: Optional[str] = None
    goals: str

    @validator("income", "savings")
    def check_numeric(cls, v):
        if not v.replace(",", "").isdigit():
            raise ValueError("Must be a numeric value (commas allowed)")
        return v

    @validator("credit_score", "lump_sum", "monthly_contribution", pre=True, always=True)
    def check_numeric_optional(cls, v):
        if v is None or v == "":
            return None
        if not v.replace(",", "").isdigit():
            raise ValueError("Must be a numeric value (commas allowed) or empty")
        return v

###################################################################################################
# AI AGENTS SETUP (Improved System Messages)
###################################################################################################

config_list = autogen.config_list_from_json("oai_config_list.json")

def get_llm_config(config_list):
    return {
        "config_list": config_list,
        "temperature": 0.7,
        # "use_cache": False  # Disable caching during development -- REMOVE THIS LINE
    }
# --- Agent Definitions (Simplified Prompts, No Markdown) ---

lead_qualification_agent = autogen.AssistantAgent(
    name="LeadQualificationAgent",
    llm_config=get_llm_config(config_list),
    system_message="""Lead Qualification Expert: Assess lead priority (high/low) and provide reasoning.
    Output JSON: {"priority": "high/low", "reasoning": "reason"}"""
)

financial_strategy_agent = autogen.AssistantAgent(
    name="FinancialStrategyAgent",
    llm_config=get_llm_config(config_list),
    system_message="""Financial Strategist: Provide budget, investment, and savings advice.
    Output JSON: {"budget": "advice", "investment": "advice", "savings": "advice"}"""
)

policy_advisor_agent = autogen.AssistantAgent(
    name="PolicyAdvisorAgent",
    llm_config=get_llm_config(config_list),
    system_message="""Insurance Policy Expert: Recommend a life insurance policy.
    Output JSON: {"policy_type": "policy type", "reasoning": "reason"}"""
)

anti_iul_agent = autogen.AssistantAgent(
    name="AntiIULAgent",
    llm_config=get_llm_config(config_list),
    system_message="""IUL/VUL Skeptic: Critique the PolicyAdvisorAgent's recommendation, highlighting risks.
    Output JSON: {"critique": "critique"}"""
)

risk_assessment_agent = autogen.AssistantAgent(
    name="RiskAssessmentAgent",
    llm_config=get_llm_config(config_list),
    system_message="""Risk Assessment Expert: Assess the client's risk tolerance (low/moderate/high).
    Consider their lump sum and monthly contributions in addition to income, savings, and goals.
    Output JSON: {"risk_tolerance": "low/moderate/high", "reasoning": "reasoning"}"""
)

followup_agent = autogen.AssistantAgent(
    name="FollowupAgent",
    llm_config=get_llm_config(config_list),
    system_message="""Follow-up Email Specialist: Draft a professional follow-up email.
    Output JSON: {"subject": "subject", "body": "body"}"""
)

################################################################################################
# AI QUALIFICATION & ANALYSIS PIPELINE (Refactored for Clarity and Robustness)
################################################################################################
async def process_lead_pipeline(lead: Lead):
    try:
        # --- 1. Data Validation (Handled by Pydantic) ---

        # --- 2. Initiate Autogen Chat ---
        user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            code_execution_config=False,
            default_auto_reply="OK.",  # Autogen requires a default auto-reply
            is_termination_msg=lambda x: x.get("content", "").strip().lower().endswith("terminate") # Changed termination message
        )

        # Define the GroupChat
        groupchat = autogen.GroupChat(
            agents=[lead_qualification_agent, financial_strategy_agent, risk_assessment_agent, policy_advisor_agent, anti_iul_agent, followup_agent],
            messages=[],
            max_round=7,  # Max rounds *per speaker*
            speaker_selection_method="round_robin"
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=get_llm_config(config_list))
        #Important:
        manager.max_round = 1 # Only one round of conversation

        # Construct the initial message
        initial_message = (
            f"Process the following lead and generate structured responses as follows:\n"
            f"- LeadQualificationAgent: Assess lead priority (high/low) and provide reasoning.\n"
            f"- FinancialStrategyAgent: Provide budgeting, investment, and savings strategies.\n"
            f"- RiskAssessmentAgent: Evaluate risk tolerance, considering lump sum/monthly contributions.\n"
            f"- PolicyAdvisorAgent: Recommend the most suitable life insurance policy.\n"
            f"- AntiIULAgent: Critique any IUL or VUL policies.\n"
            f"- FollowupAgent: Draft a follow-up email.\n\n"
            f"Lead Details:\n"
            f"Name: {lead.name}\n"
            f"Income: {lead.income}\n"
            f"Savings: {lead.savings}\n"
            f"Credit Score: {lead.credit_score if lead.credit_score else 'N/A'}\n"
            f"Date of Birth/Age: {lead.dob if lead.dob else 'N/A'}\n"
            f"Lump Sum Contribution: {lead.lump_sum if lead.lump_sum else 'N/A'}\n"
            f"Monthly Contribution: {lead.monthly_contribution if lead.monthly_contribution else 'N/A'}\n"
            f"Goals: {lead.goals}\n\n"
            "TERMINATE"  # Add termination message
        )

        # Start the chat
        await user_proxy.a_initiate_chat(manager, message=initial_message)
        # --- 3. Extract and Process Responses ---
        results: Dict[str, Any] = {}  # Use a dictionary to store results

        logger.debug("groupchat.messages: %s", groupchat.messages)

        for message in groupchat.messages:
            # Use message['name'] to identify the agent, NOT message['role']
            if message['role'] == 'user' and message['name'] != 'User':  # Corrected extraction logic
                agent_name = message['name']
                content = message.get('content', '')
                logger.debug(f"Processing message from {agent_name}: {content}")

                try:
                    # More Robust JSON Parsing: Check if content looks like JSON
                    parsed_content = json.loads(content) if content.strip().startswith("{") and content.strip().endswith("}") else {}
                    if parsed_content:
                        results[agent_name] = parsed_content
                    else:
                        logger.warning(f"Agent {agent_name} returned malformed JSON.")
                        results[agent_name] = {"error": "Malformed JSON from AI"}
                except json.JSONDecodeError:
                    logger.error(f"JSONDecodeError for {agent_name}. Content: {content}")
                    results[agent_name] = {"error": "Invalid JSON from AI"}  # More informative error

        # --- 4. Combine Results (with Fallbacks) ---
        ai_analysis = {
            "qualification": results.get("LeadQualificationAgent", {"priority": "N/A", "reasoning": "N/A"}),
            "financial_strategy": results.get("FinancialStrategyAgent", {"budget": "N/A", "investment": "N/A", "savings": "N/A"}),
            "policy": results.get("PolicyAdvisorAgent", {"policy_type": "N/A", "reasoning": "N/A"}),
            "anti_iul_critique": results.get("AntiIULAgent", {"critique": "N/A"}),
            "risk_assessment": results.get("RiskAssessmentAgent", {"risk_tolerance": "N/A", "reasoning": "N/A"}),
            "followup": results.get("FollowupAgent", {"subject": "N/A", "body": "N/A"}),
        }
        logger.debug("Final ai_analysis: %s", ai_analysis)

        # --- 5. Store Results (Google Sheets) ---
        sheet.append_row([
            lead.name,
            lead.income,
            lead.savings,
            lead.credit_score if lead.credit_score else "N/A",
            lead.dob if lead.dob else "N/A",
            lead.lump_sum if lead.lump_sum else "N/A",
            lead.monthly_contribution if lead.monthly_contribution else "N/A",
            lead.goals,
            json.dumps(ai_analysis)  # Store the ENTIRE ai_analysis as a JSON string
        ])

        # --- 6. Return Results ---
        return {"status": "success", "message": "Lead processed successfully!", "ai_analysis": json.dumps(ai_analysis)}

    except HTTPException as e:
        return {"status": "error", "message": str(e.detail)}
    except Exception as e:
        error_message = f"Error processing lead: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return {"status": "error", "message": "Internal Server Error."}

####################################################################################################
# API ENDPOINT
####################################################################################################

@app.post("/process-lead")
async def process_lead(lead: Lead):
    return await process_lead_pipeline(lead)