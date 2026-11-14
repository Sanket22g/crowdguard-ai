import asyncio
import httpx
import logging
from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langsmith import traceable
from app.config import Config
from app.retry import with_retry

logger = logging.getLogger(__name__)


class VettingState(TypedDict):
    topic: str                   # NGO campaign proposal / grant submission text
    session_id: str              # Unique audit session identifier
    session_history: list[dict]  # Live dialogue between auditor and applicant
    ltm_context: str             # Past similar campaign / scam embeddings from pgvector
    search_results: list[str]    # Document & credential audit findings
    summaries: list[str]         # Forensic fraud signals & budget realism assessment
    report: str                  # Official CrowdGuard CSR Due Diligence Report
    verified: bool               # Compliance officer verification approval flag
    error: str
    iterations: int


# Backwards compatibility alias
ResearchState = VettingState


async def _tz_call(config: Config, function_name: str, message: str) -> str:
    return await with_retry(
        lambda: _tz_call_once(config, function_name, message),
        max_retries=config.llm_max_retries,
        delay=config.llm_retry_delay,
    )


async def _tz_call_once(config: Config, function_name: str, message: str) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{config.tensorzero_url}/inference",
            json={
                "function_name": function_name,
                "input": {"messages": [{"role": "user", "content": message}]},
            },
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]


class DocumentAuditorAgent:
    """Document & Credential Auditor: Extracts organization details, registration IDs, budget math, and core claims."""

    def __init__(self, config: Config):
        self.config = config

    @traceable(run_type="tool", name="agent:document_auditor")
    async def run(self, topic: str, session_history: list[dict]) -> str:
        logger.info(f"DocumentAuditorAgent: auditing campaign/NGO submission '{topic[:60]}'")

        history_ctx = ""
        if session_history:
            recent = session_history[-4:]  # last 4 turns for context
            history_ctx = "\n\nPrevious conversation context:\n"
            history_ctx += "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

        return await _tz_call(
            self.config,
            "research_summarize",
            f"You are CrowdGuard AI — Senior CSR & Charity Due Diligence Auditor for Crowdera.\n"
            f"Thoroughly analyze this NGO / Crowdfunding campaign proposal:\n"
            f"'''\n{topic}\n'''\n\n"
            f"Extract and verify:\n"
            f"1. Organization Name & Claimed Mission\n"
            f"2. Registration / Tax ID (e.g. UEN, 80G, 501c3) — check if format looks valid or missing\n"
            f"3. Requested Funding Amount & Target Beneficiaries\n"
            f"4. Itemized Budget Breakdown: Calculate the % of total funds allocated to Administrative / Overhead costs.\n"
            f"Be precise, objective, and highlight any missing verification credentials.{history_ctx}",
        )


class FraudInvestigatorAgent:
    """Fraud & Risk Investigator: Detects red flags, inflated budgets, emotional manipulation, and AML risks."""

    def __init__(self, config: Config):
        self.config = config

    @traceable(run_type="tool", name="agent:fraud_investigator")
    async def run(self, search_results: list[str]) -> str:
        logger.info("FraudInvestigatorAgent: analyzing fraud signals and risk indicators")
        combined = "\n\n".join(search_results)
        return await _tz_call(
            self.config,
            "research_summarize",
            f"You are CrowdGuard AI's Financial Crime & Fraud Investigator for Corporate Philanthropy and Crowdera.\n"
            f"Review these audit findings and identify all risk indicators:\n\n{combined}\n\n"
            f"Specifically evaluate:\n"
            f"- Budget Realism: Is administrative/overhead expense >15%? (High overhead is a major scam indicator).\n"
            f"- Operational Feasibility: Are cost-per-beneficiary metrics reasonable?\n"
            f"- Red Flags: Any vague expense categories, unverified personal bank transfers, or pressure tactics?\n"
            f"Summarize your risk assessment into clear, structured bullet points.",
        )


class VettingReportAgent:
    """
    CSR Due Diligence Report Writer: Produces the final structured Trust Score and Vetting Audit.
    Leverages past campaign memory from pgvector to identify repeat copy-paste campaigns.
    """

    def __init__(self, config: Config):
        self.config = config

    @traceable(run_type="tool", name="agent:vetting_writer")
    async def run(self, topic: str, summaries: list[str], ltm_context: str) -> str:
        logger.info("VettingReportAgent: drafting CSR Due Diligence Report")
        combined = "\n\n".join(summaries)

        ltm_section = ""
        if ltm_context:
            ltm_section = (
                f"\n\nHISTORICAL DATABASE MATCH (Similar past campaigns in vector memory):\n"
                f"{ltm_context[:2000]}\n"
                f"Note: If the current submission is nearly identical to a past flagged campaign, call this out as potential plagiarism/scam!"
            )

        return await _tz_call(
            self.config,
            "report_write",
            f"Generate an official CrowdGuard AI CSR & NGO Due Diligence Report for Crowdera and enterprise donors.\n\n"
            f"Campaign Submission:\n{topic[:1000]}\n\n"
            f"Investigative Risk Findings:\n{combined}\n"
            f"{ltm_section}\n\n"
            f"Structure your report exactly as follows:\n"
            f"1. EXECUTIVE SUMMARY: Purpose, funding requested, and organization overview.\n"
            f"2. CROWDGUARD TRUST SCORE (0 to 100): Provide a numeric score based on financial transparency and compliance.\n"
            f"3. KEY FINDINGS & BUDGET ANALYSIS: Breakdown of costs, admin fees %, and beneficiary metrics.\n"
            f"4. RISK FACTORS & RED FLAGS: Specific concerns regarding fraud, credentials, or feasibility.\n"
            f"5. FINAL RECOMMENDATION: Explicitly state: [APPROVED / FLAGGED FOR MANUAL REVIEW / REJECTED] with concise justification.",
        )


class ComplianceCriticAgent:
    """Compliance Officer: Validates that the report is rigorous, logically consistent, and factually backed."""

    def __init__(self, config: Config):
        self.config = config

    @traceable(run_type="tool", name="agent:compliance_critic")
    async def run(self, report: str) -> bool:
        logger.info("ComplianceCriticAgent: reviewing vetting report standards")
        check = await _tz_call(
            self.config,
            "research_summarize",
            f"You are the CrowdGuard AI Chief Compliance Officer. Review this NGO Vetting Report.\n"
            f"Does it contain a clear CrowdGuard Trust Score (0-100), explicit budget analysis, and an unambiguous recommendation?\n"
            f"Reply with YES if it meets institutional audit standards, or NO with reason:\n\n"
            f"{report[:self.config.agent_report_truncate]}",
        )
        return check.strip().upper().startswith("YES")


# Backwards compatibility aliases
SearchAgent = DocumentAuditorAgent
SummarizeAgent = FraudInvestigatorAgent
WriterAgent = VettingReportAgent
CriticAgent = ComplianceCriticAgent


class OrchestratorAgent:
    """
    Coordinates all sub-agents. Passes session history to DocumentAuditorAgent and LTM context
    to VettingReportAgent so the pipeline is genuinely context-aware. If ComplianceCriticAgent rejects
    the report it loops back for another pass (up to agent_max_iterations times).
    """

    def __init__(self, config: Config):
        self.config = config
        self.auditor_agent = DocumentAuditorAgent(config)
        self.investigator_agent = FraudInvestigatorAgent(config)
        self.writer_agent = VettingReportAgent(config)
        self.critic_agent = ComplianceCriticAgent(config)

    @traceable(run_type="chain", name="orchestrator:auditor")
    async def search_node(self, state: VettingState) -> dict:
        result = await self.auditor_agent.run(state["topic"], state.get("session_history", []))
        return {"search_results": [result]}

    @traceable(run_type="chain", name="orchestrator:investigator")
    async def summarize_node(self, state: VettingState) -> dict:
        summary = await self.investigator_agent.run(state["search_results"])
        return {"summaries": [summary]}

    @traceable(run_type="chain", name="orchestrator:writer")
    async def write_node(self, state: VettingState) -> dict:
        report = await self.writer_agent.run(
            state["topic"],
            state["summaries"],
            state.get("ltm_context", ""),
        )
        return {"report": report, "iterations": state.get("iterations", 0) + 1}

    @traceable(run_type="chain", name="orchestrator:compliance")
    async def verify_node(self, state: VettingState) -> dict:
        verified = await self.critic_agent.run(state["report"])
        return {"verified": verified}

    def route(self, state: VettingState) -> str:
        """Compliance decision: retry audit or approve final report."""
        if not state["verified"] and state.get("iterations", 0) < self.config.agent_max_iterations:
            logger.info(f"Compliance critic rejected report — re-auditing (iteration {state['iterations']})")
            return "audit_documents"
        return END


def build_graph(config: Config):
    orchestrator = OrchestratorAgent(config)
    workflow = StateGraph(VettingState)

    workflow.add_node("audit_documents", orchestrator.search_node)
    workflow.add_node("investigate_fraud", orchestrator.summarize_node)
    workflow.add_node("draft_due_diligence_report", orchestrator.write_node)
    workflow.add_node("compliance_verification", orchestrator.verify_node)

    workflow.add_edge(START, "audit_documents")
    workflow.add_edge("audit_documents", "investigate_fraud")
    workflow.add_edge("investigate_fraud", "draft_due_diligence_report")
    workflow.add_edge("draft_due_diligence_report", "compliance_verification")
    workflow.add_conditional_edges(
        "compliance_verification",
        orchestrator.route,
        {"audit_documents": "audit_documents", END: END},
    )

    return workflow.compile()
