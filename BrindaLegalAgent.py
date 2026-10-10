import os
from openai import OpenAI
from dotenv import load_dotenv
import threading

# Load environment variables and initialize OpenAI client
load_dotenv()
client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Shared dict for thread-safe collection of agent outputs
agent_outputs = {}

# Example contract text
contract_text = """
CONSULTING AGREEMENT

This Consulting Agreement (the "Agreement") is made effective as of January 1, 2025 (the "Effective Date"), by and between ABC Corporation, a Delaware corporation ("Client"), and XYZ Consulting LLC, a California limited liability company ("Consultant").

1. SERVICES. Consultant shall provide Client with the following services: strategic business consulting, market analysis, and technology implementation advice (the "Services").

2. TERM. This Agreement shall commence on the Effective Date and shall continue for a period of 12 months, unless earlier terminated.

3. COMPENSATION. Client shall pay Consultant a fee of $10,000 per month for Services rendered. Payment shall be made within 30 days of receipt of Consultant's invoice.

4. CONFIDENTIALITY. Consultant acknowledges that during the engagement, Consultant may have access to confidential information. Consultant agrees to maintain the confidentiality of all such information.

5. INTELLECTUAL PROPERTY. All materials developed by Consultant shall be the property of Client. Consultant assigns all right, title, and interest in such materials to Client.

6. TERMINATION. Either party may terminate this Agreement with 30 days' written notice. Client shall pay Consultant for Services performed through the termination date.

7. GOVERNING LAW. This Agreement shall be governed by the laws of the State of Delaware.

8. LIMITATION OF LIABILITY. Consultant's liability shall be limited to the amount of fees paid by Client under this Agreement.

9. INDEMNIFICATION. Client shall indemnify Consultant against all claims arising from use of materials provided by Client.

10. ENTIRE AGREEMENT. This Agreement constitutes the entire understanding between the parties and supersedes all prior agreements.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first above written.
"""


# ---------------------------------------------------------
# Specialist Agent 1
# ---------------------------------------------------------

class LegalTermsChecker:
    """Agent that checks for problematic legal terms and clauses in contracts."""

    def run(self, contract_text):
        print("LegalTermsChecker: Analyzing legal terms...")

        system_prompt = """
You are a legal expert specializing in contract law.

Review the provided contract and identify:
1. Potentially problematic or ambiguous clauses
2. Missing or unclear legal protections
3. Unusual or one-sided provisions
4. Important termination, liability, indemnification,
   confidentiality, intellectual property, and governing-law issues
5. Any language that may require clarification

Provide your findings clearly and concisely. Do not invent facts
that are not present in the contract.
"""

        user_prompt = f"""
Please analyze the following contract for legal terms and clauses:

{contract_text}
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        agent_outputs["legal"] = response.choices[0].message.content


# ---------------------------------------------------------
# Specialist Agent 2
# ---------------------------------------------------------

class ComplianceValidator:
    """Agent that validates regulatory and industry compliance of contracts."""

    def run(self, contract_text):
        print("ComplianceValidator: Checking compliance...")

        system_prompt = """
You are a contract compliance specialist.

Review the provided contract for potential regulatory,
industry-standard, and compliance concerns.

Consider areas such as:
1. Data privacy and protection
2. Confidentiality requirements
3. Intellectual property obligations
4. Employment or contractor classification concerns
5. Regulatory obligations that may apply
6. Missing compliance provisions
7. Industry-standard contractual protections

Clearly distinguish between actual issues stated in the contract
and areas that may require further legal or regulatory review.

Do not assume a specific industry regulation applies unless the
contract provides enough information to support that conclusion.
"""

        user_prompt = f"""
Please review the following contract for compliance concerns:

{contract_text}
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        agent_outputs["compliance"] = response.choices[0].message.content


# ---------------------------------------------------------
# Specialist Agent 3
# ---------------------------------------------------------

class FinancialRiskAssessor:
    """Agent that assesses financial risks and liabilities in contracts."""

    def run(self, contract_text):
        print("FinancialRiskAssessor: Evaluating financial risks...")

        system_prompt = """
You are a financial risk specialist who reviews commercial contracts.

Analyze the provided contract for financial risks and liabilities.

Focus on:
1. Payment terms and financial obligations
2. Pricing and recurring costs
3. Termination-related financial exposure
4. Liability limitations
5. Indemnification obligations
6. Potential unexpected costs
7. Financial exposure for each party
8. Missing financial protections or unclear payment provisions

Identify the most important financial risks and explain their
potential business impact.

Do not invent financial information that is not contained in
the contract.
"""

        user_prompt = f"""
Please assess the following contract for financial risks:

{contract_text}
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        agent_outputs["financial"] = response.choices[0].message.content


# ---------------------------------------------------------
# Summary Agent
# ---------------------------------------------------------

class SummaryAgent:
    """Agent that synthesizes findings from all specialized agents."""

    def run(self, contract_text, inputs):
        print("SummaryAgent: Synthesizing all findings...")

        legal_findings = inputs.get(
            "legal",
            "No legal analysis provided."
        )

        compliance_findings = inputs.get(
            "compliance",
            "No compliance analysis provided."
        )

        financial_findings = inputs.get(
            "financial",
            "No financial analysis provided."
        )

        system_prompt = """
You are a senior legal counsel reviewing a commercial contract.

You have received analyses from three specialist agents:
- Legal terms specialist
- Compliance specialist
- Financial risk specialist

Your task is to synthesize their findings into one clear,
comprehensive executive summary.

Your summary should:
1. Identify the most important issues
2. Highlight significant legal, compliance, and financial risks
3. Explain the potential business impact
4. Identify areas requiring further legal review
5. Provide an overall assessment of the contract

Do not invent facts or introduce issues that are unsupported by
the contract or specialist analyses.

This is an analytical review, not formal legal advice.
"""

        user_prompt = f"""
Please synthesize the following contract analysis into a
comprehensive executive summary.

ORIGINAL CONTRACT:
{contract_text}

LEGAL TERMS ANALYSIS:
{legal_findings}

COMPLIANCE VALIDATION:
{compliance_findings}

FINANCIAL RISK ASSESSMENT:
{financial_findings}

Provide a consolidated executive summary with:
- Overall assessment
- Key legal concerns
- Key compliance concerns
- Key financial concerns
- Recommended areas for further review
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content


# ---------------------------------------------------------
# Orchestration Function
# ---------------------------------------------------------

def analyze_contract(contract_text):
    """Run all agents in parallel and summarize their findings."""

    # Clear outputs in case analyze_contract is called more than once
    agent_outputs.clear()

    # 1. Create agent instances
    legal_checker = LegalTermsChecker()
    compliance_validator = ComplianceValidator()
    financial_assessor = FinancialRiskAssessor()
    summary_creator = SummaryAgent()

    # 2. Create threads for specialist agents
    threads = [
        threading.Thread(
            target=legal_checker.run,
            args=(contract_text,)
        ),
        threading.Thread(
            target=compliance_validator.run,
            args=(contract_text,)
        ),
        threading.Thread(
            target=financial_assessor.run,
            args=(contract_text,)
        )
    ]

    # 3. Start all specialist agents
    for thread in threads:
        thread.start()

    # 4. Wait for all specialist agents to finish
    for thread in threads:
        thread.join()

    # 5. Generate final summary
    final_report = summary_creator.run(
        contract_text,
        agent_outputs
    )

    # 6. Return final analysis
    return final_report


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Enterprise Contract Analysis System")
    print("Analyzing contract...")

    final_analysis = analyze_contract(contract_text)

    print("\n=== FINAL CONTRACT ANALYSIS ===\n")
    print(final_analysis)