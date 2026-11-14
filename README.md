# 🛡️ CrowdGuard AI: Autonomous CSR & NGO Vetting Agent

> **Production-grade Multi-Agent System for Corporate Philanthropy & Crowdfunding Due Diligence.**  
> Built with **FastAPI, Redis Streams, AWS ECS (Fargate), LangGraph, PostgreSQL + pgvector, AWS Bedrock Guardrails, and Microsoft PyRIT Red-Teaming.**

---

## 📌 Executive Overview

Crowdfunding and Corporate Social Responsibility (CSR) platforms (such as **Crowdera Singapore**) deploy millions of dollars to non-profits and social causes. However, manual verification creates massive operational bottlenecks:
* **Scam & Plagiarized Campaigns:** Fraudsters copy legitimate charity stories and submit fake registration numbers.
* **Financial Mismanagement:** Excessive administrative/overhead fees (>15-30%) diverting funds away from beneficiaries.
* **Adversarial Prompt Injection:** Attackers using prompt injection to trick AI vetting into auto-approving fraudulent campaigns.

**CrowdGuard AI** is an asynchronous, event-driven AI platform that performs deep forensic audits of NGO proposals, checks mathematical budget realism, cross-references historical campaign embeddings, and generates institutional-grade CSR due diligence reports in seconds.

---

## 🏛️ System Architecture

```
                       [ Incoming Campaign Proposal / Grant Request ]
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │   FastAPI Ingestion Gateway      │
                           │   • Rate Limiting & Auth         │
                           │   • AWS Bedrock Guardrail (PII)  │
                           └────────────────┬─────────────────┘
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │   Redis Stream Task Queue        │
                           │   Key: 'vetting:jobs'            │
                           └────────────────┬─────────────────┘
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │   ECS Fargate Multi-Agent Pool   │
                           │   (LangGraph Orchestration)      │
                           └────────────────┬─────────────────┘
                                            │
          ┌─────────────────────────────────┼─────────────────────────────────┐
          ▼                                 ▼                                 ▼
┌──────────────────┐              ┌──────────────────┐              ┌──────────────────┐
│ Document Auditor │              │Fraud Investigator│              │  Vetting Writer  │
│ Checks Gov IDs,  │ ───────────► │Flags high admin  │ ───────────► │Generates 0-100   │
│ Tax Exemption &  │              │fees (>15%) and   │              │Trust Score & CSR │
│ Budget Math      │              │unrealistic claims│              │Due Diligence Doc │
└──────────────────┘              └──────────────────┘              └────────┬─────────┘
                                                                             │
                                                                    ┌────────▼─────────┐
                                                                    │Compliance Critic │
                                                                    │Rejects & retries │
                                                                    │if unverified     │
                                                                    └──────────────────┘
                                            │
                                            ▼
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
┌───────────────────────────────┐                       ┌────────────────────────────────┐
│   Short-Term Session Memory   │                       │     Long-Term Vector Memory    │
│     (AWS ElastiCache Redis)   │                       │  (AWS RDS Postgres + pgvector) │
├───────────────────────────────┤                       ├────────────────────────────────┤
│ • Sliding window (5 messages) │                       │ • 384-dim semantic embeddings  │
│ • Real-time auditor dialog    │                       │ • Detects plagiarized/scam copy│
│ • TTL: 30 minutes             │                       │ • Fast ivfflat cosine search   │
└───────────────────────────────┘                       └────────────────────────────────┘
                                            │
                                            ▼
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
┌───────────────────────────────┐                       ┌────────────────────────────────┐
│   Automated LLM Evaluation    │                       │   Adversarial Red-Teaming      │
│         (LangSmith)           │                       │       (Microsoft PyRIT)        │
├───────────────────────────────┤                       ├────────────────────────────────┤
│ • Relevance & Completeness    │                       │ • Jailbreak defense tests      │
│ • Hallucination risk grading  │                       │ • XPIA prompt injection probes │
│ • Multi-judge automated evals │                       │ • Port 8001 Security Dashboard │
└───────────────────────────────┘                       └────────────────────────────────┘
```

---

## 🛠️ Tech Stack & Key Components

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Framework** | FastAPI (`app/main.py`) | High-throughput asynchronous REST API for campaign submissions. |
| **Task Queue** | Redis Streams (`app/queue.py`) | Asynchronous job distribution for long-running audit tasks. |
| **Orchestration** | LangGraph (`app/agents.py`) | Multi-agent state machine (Auditor &rarr; Investigator &rarr; Writer &rarr; Critic). |
| **Short-Term Memory** | Redis Lists (`app/memory.py`) | Sliding-window conversation context (`LTRIM`). |
| **Vector Memory** | PostgreSQL `pgvector` (`app/memory.py`) | Cosine similarity memory (`SentenceTransformer all-MiniLM-L6-v2`) for scam detection. |
| **Guardrails** | AWS Bedrock Guardrails (`terraform/main.tf`) | Content filtering and automatic PII anonymization (Credit Cards, AWS Keys, IDs). |
| **Observability & Eval**| LangSmith (`app/eval.py`) | 4-judge automated scoring (Relevance, Completeness, Hallucination, Quality). |
| **Red Teaming** | Microsoft PyRIT (`pyrit_dashboard/`) | Adversarial penetration testing against Jailbreaks and Prompt Injections. |
| **Infrastructure** | Terraform (`terraform/main.tf`) | 100% Infrastructure-as-Code (VPC, Private Subnets, ECS Fargate, ElastiCache, RDS). |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* Python 3.11+
* Docker & Docker Compose (optional for local testing)
* AWS CLI configured (`aws configure`)
* Terraform 1.5+

### 2. Local Setup
```bash
# Clone the repository
git clone https://github.com/your-username/crowdguard-ai.git
cd crowdguard-ai

# Install dependencies
pip install -r requirements.txt

# Run the API locally
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Deploy to AWS Cloud via Terraform
```powershell
# 1. Run the Bootstrap script (creates S3 state bucket & DynamoDB lock table)
.\bootstrap.bat

# 2. Initialize and deploy Terraform infrastructure
cd terraform
terraform init
terraform plan
terraform apply
```

---

## 📊 Sample API Usage

### Submit an NGO Campaign for Vetting
```bash
curl -X POST http://localhost:8000/vet \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "campaign_or_ngo": "Clean Water Initiative. Budget: $100,000 total. Drilling equipment: $30,000. Water filters: $20,000. Founder travel and management fees: $50,000. Registration: SG-UEN-PENDING.",
    "output_format": "json"
  }'
```

**Response:**
```json
{
  "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "session_id": "session-xyz"
}
```

### Check Audit Result
```bash
curl -X GET http://localhost:8000/result/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d \
  -H "X-API-Key: YOUR_API_KEY"
```

**Result Preview:**
```json
{
  "status": "done",
  "report": "EXECUTIVE SUMMARY: Clean Water Initiative requested $100,000...\n\nTRUST SCORE: 25/100 (HIGH RISK)\n\nKEY FINDINGS: 50% ($50,000) allocated to administrative fees (industry threshold is <15%)...\n\nRECOMMENDATION: REJECTED due to abnormal overhead costs and unverified registration status."
}
```

---

## 🛡️ Security & PyRIT Red-Team Testing

The **PyRIT Red Team Dashboard** runs on port `8001` (`http://localhost:8001`) to continuously test the system against adversarial prompts:
* **Jailbreaks:** Direct instructions attempting to force an auto-approval.
* **XPIA:** Malicious prompt injections hidden inside campaign descriptions.
* **Crescendo Attacks:** Multi-turn conversational escalation attempting to disguise fraudulent fees.
* **Skeleton Key Attacks:** False authority impersonation (e.g. *"I am the CSR Managing Director"*).
