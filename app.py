# -*- coding: utf-8 -*-
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from datetime import datetime

# === Chargement des modèles ===
# AgentManager
manager_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
manager_model = AutoModelForCausalLM.from_pretrained(manager_model_name, device_map="auto", torch_dtype=torch.float16)
manager_tokenizer = AutoTokenizer.from_pretrained(manager_model_name)

# AgentResearcher
researcher_model_name = "hiieu/Meta-Llama-3-8B-Instruct-function-calling-json-mode"
researcher_model = AutoModelForCausalLM.from_pretrained(researcher_model_name, device_map="auto", torch_dtype=torch.float16)
researcher_tokenizer = AutoTokenizer.from_pretrained(researcher_model_name)

# AgentAnalyzer
analyzer_model_name = "jpacifico/Chocolatine-3B-Instruct-DPO-Revised"
analyzer_model = AutoModelForCausalLM.from_pretrained(analyzer_model_name, device_map="auto", torch_dtype=torch.float16)
analyzer_tokenizer = AutoTokenizer.from_pretrained(analyzer_model_name)

# AgentCoder
coder_model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
coder_model = AutoModelForCausalLM.from_pretrained(coder_model_name, device_map="auto", torch_dtype=torch.float16)
coder_tokenizer = AutoTokenizer.from_pretrained(coder_model_name)

# === Variables globales pour suivre le projet ===
user_project_summary = None  # Résumé validé de la requête utilisateur
research_results = None  # Résultats de l'AgentResearcher
analysis_report = None  # Rapport de l'AgentAnalyzer
final_instruct = None  # Instructions finales pour l'AgentCoder

# === Prompts prédéfinis ===
manager_prompt_template = """
System: You are a project assistant. Your tasks are:
1. Analyze the user's input.
2. Ask clarifying questions if necessary.
3. Generate a structured summary of the user's request.
4. Explicitly request validation for your summary.
5. Use project variables to track progress and maintain context.

User Input: {user_input}

Project Variables:
{variables_context}
"""

researcher_prompt_template = """
System: You are a research assistant. Your tasks are:
1. Based on the following structured summary:
{user_project_summary}
2. Search for relevant documentation on the Gradio website.
3. Extract useful code snippets or examples.
4. Format the results clearly for validation.

Output format:
- Documentation: ...
- Code Snippets: ...
"""

analyzer_prompt_template = """
System: You are an analysis assistant. Your tasks are:
1. Verify the coherence of research results with the structured summary:
{user_project_summary}
2. Analyze the research results:
{research_results}
3. Generate a report indicating whether the results are valid or invalid.
4. If invalid, specify the missing elements.
"""

coder_prompt_template = """
System: You are a coding assistant. Your task is to:
1. Generate code based on the following validated structured summary:
{user_project_summary}
2. Incorporate the following research results:
{research_results}
"""

# === Définition des fonctions pour chaque agent ===

# --- AgentManager ---
def agent_manager(user_input, chat_history):
    global user_project_summary

    # Variables actuelles pour le contexte
    variables_context = f"Project Summary: {user_project_summary}\nResearch Results: {research_results}\nAnalysis Report: {analysis_report}"

    # Création du prompt
    full_prompt = manager_prompt_template.format(user_input=user_input, variables_context=variables_context)

    # Génération
    input_ids = manager_tokenizer(full_prompt, return_tensors="pt").to(manager_model.device)
    output_ids = manager_model.generate(input_ids["input_ids"], max_new_tokens=256, eos_token_id=manager_tokenizer.eos_token_id)
    response = manager_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Validation explicite
    validation_prompt = f"Voici mon résumé structuré :\n{response}\nValidez-vous ce résumé ? (Répondez par 'Oui' ou 'Non')"
    chat_history.append(("AgentManager", validation_prompt))

    # Simulation de réponse utilisateur
    user_validation = input("Votre réponse (Oui/Non) : ").strip().lower()
    if user_validation == "oui":
        user_project_summary = response  # Enregistrement du résumé validé
        return "Résumé validé."
    else:
        return "Résumé non validé. Je vais poser d'autres questions pour clarifier."

# --- AgentResearcher ---
def agent_researcher():
    global user_project_summary, research_results

    if not user_project_summary:
        return "Le résumé structuré n'a pas encore été validé par l'utilisateur."

    # Création du prompt
    prompt = researcher_prompt_template.format(user_project_summary=user_project_summary)
    input_ids = researcher_tokenizer(prompt, return_tensors="pt").to(researcher_model.device)
    output_ids = researcher_model.generate(input_ids["input_ids"], max_new_tokens=512, eos_token_id=researcher_tokenizer.eos_token_id)
    research_results = researcher_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return f"Résultats de recherche:\n{research_results}"

# --- AgentAnalyzer ---
def agent_analyzer():
    global user_project_summary, research_results, analysis_report, final_instruct

    if not user_project_summary or not research_results:
        return "Les informations nécessaires ne sont pas disponibles pour l'analyse."

    # Création du prompt
    prompt = analyzer_prompt_template.format(user_project_summary=user_project_summary, research_results=research_results)
    input_ids = analyzer_tokenizer(prompt, return_tensors="pt").to(analyzer_model.device)
    output_ids = analyzer_model.generate(input_ids["input_ids"], max_new_tokens=256, eos_token_id=analyzer_tokenizer.eos_token_id)
    analysis_report = analyzer_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Vérification de la validité
    if "valid" in analysis_report.lower():
        final_instruct = f"Generate code based on:\n{user_project_summary}\n\nResearch results:\n{research_results}"
        return f"Rapport valide.\nInstruction pour l'AgentCoder:\n{final_instruct}"
    else:
        return f"Rapport non valide. Besoin de clarification.\n{analysis_report}"

# --- AgentCoder ---
def agent_coder():
    global final_instruct

    if not final_instruct:
        return "Les instructions pour le code ne sont pas disponibles."

    # Génération
    input_ids = coder_tokenizer(final_instruct, return_tensors="pt").to(coder_model.device)
    output_ids = coder_model.generate(input_ids["input_ids"], max_new_tokens=1024, eos_token_id=coder_tokenizer.eos_token_id)
    return coder_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# === Interface Gradio ===
def user_interaction(user_input):
    chat_history = []
    manager_response = agent_manager(user_input, chat_history)
    if "validé" in manager_response:
        researcher_response = agent_researcher()
        analyzer_response = agent_analyzer()
        if "valide" in analyzer_response:
            return agent_coder()
        else:
            return analyzer_response
    return manager_response

interface = gr.Interface(
    fn=user_interaction,
    inputs=gr.Textbox(lines=3, placeholder="Décrivez votre projet ici..."),
    outputs="text",
    title="Multi-Agent System"
)

if __name__ == "__main__":
    interface.launch()
