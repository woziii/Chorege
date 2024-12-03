# -*- coding: utf-8 -*-

# === Imports ===
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from datetime import datetime
import os
import json
import logging

# --- Imports spécifiques pour l'AgentResearcher ---
import requests
from bs4 import BeautifulSoup
import logging
from concurrent.futures import ThreadPoolExecutor

# === Configuration du logger ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("project.log"),
        logging.StreamHandler()
    ]
)

# === Chargement des modèles ===
# Chargement du modèle pour l'AgentManager
manager_model_name = "meta-llama/Llama-3.1-8B-Instruct"
manager_model = AutoModelForCausalLM.from_pretrained(
    manager_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16  # Utilisation de bfloat16 comme recommandé
)
manager_tokenizer = AutoTokenizer.from_pretrained(manager_model_name)

# AgentResearcher
researcher_model_name = "hiieu/Meta-Llama-3-8B-Instruct-function-calling-json-mode"
researcher_model = AutoModelForCausalLM.from_pretrained(
    researcher_model_name,
    torch_dtype=torch.bfloat16,  # Utilisation de bfloat16 comme recommandé
    device_map="auto",
)
researcher_tokenizer = AutoTokenizer.from_pretrained(researcher_model_name)

# AgentAnalyzer
analyzer_model_name = "jpacifico/Chocolatine-3B-Instruct-DPO-Revised"
analyzer_model = AutoModelForCausalLM.from_pretrained(
    analyzer_model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
analyzer_tokenizer = AutoTokenizer.from_pretrained(analyzer_model_name)

# AgentCoder
# AgentCoder
coder_model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
coder_model = AutoModelForCausalLM.from_pretrained(
    coder_model_name,
    torch_dtype="auto",
    device_map="auto"
)
coder_tokenizer = AutoTokenizer.from_pretrained(coder_model_name)

# === Variables Globales ===
project_state = {
    "AgentManager": {"structured_summary": None},
    "AgentResearcher": {"search_results": None},
    "AgentAnalyzer": {"analysis_report": None, "instruction_for_coder": None},
    "AgentCoder": {"final_code": None}
}

# --- Prompts prédéfinis ---
manager_prompt_template = """
Vous êtes l'AgentManager d'un système multi-agent.

- Votre rôle est d'interagir avec l'utilisateur pour comprendre sa demande.
- Vous devez poser des questions pertinentes pour obtenir toutes les informations nécessaires.
- Une fois que vous estimez avoir suffisamment d'informations, vous générez un résumé structuré du projet.
- Vous incluez les informations des variables du projet si elles ne sont pas vides.
- Vous demandez une validation explicite à l'utilisateur pour le résumé généré.
- Vous pouvez modifier les variables du projet si l'utilisateur en fait la demande.

Variables du projet :
{variables_context}
"""

researcher_prompt_template = """
System: Vous êtes un assistant de recherche. Vos tâches sont :
1. Basé sur le résumé structuré suivant :
{structured_summary}
2. Effectuer des recherches dans la documentation Gradio en ligne.
3. Extraire des extraits de code ou des exemples utiles.
4. Formater clairement les résultats pour validation.

Format de sortie :
- Documentation : ...
- Extraits de code : ...
"""

analyzer_prompt_template = """
Vous êtes un assistant d'analyse. Vos tâches sont :
1. Vérifier la cohérence des résultats de recherche avec le résumé structuré :
{structured_summary}
2. Analyser les résultats de recherche :
{search_results}
3. Générer un rapport indiquant si les résultats sont **valide** ou **non valide**.
4. Si **non valide**, spécifier les éléments manquants ou incohérences.
5. Votre réponse doit commencer par 'Validité: Oui' ou 'Validité: Non', suivi du rapport d'analyse.
"""

coder_prompt_template = """
System: Vous êtes un assistant de codage. Votre tâche est de :
1. Générer du code basé sur le résumé structuré validé suivant :
{structured_summary}
2. Incorporer les résultats de recherche suivants :
{search_results}
"""

# === Définition des fonctions pour chaque agent ===


# === Fonctions Utilitaires de l'agentManager ===
def get_variables_context():
    variables = {}
    for agent, data in project_state.items():
        variables[agent] = {}
        for key, value in data.items():
            variables[agent][key] = value if value else "N/A"
    variables_context = json.dumps(variables, indent=2, ensure_ascii=False)
    return variables_context

def update_project_state(modifications):
    for var, value in modifications.items():
        keys = var.split('.')
        target = project_state
        for key in keys[:-1]:
            target = target.get(key, {})
        target[keys[-1]] = value

def extract_modifications(user_input):
    # Extraction simplifiée pour l'exemple
    modifications = {}
    if "modifie" in user_input.lower():
        import re
        matches = re.findall(r"modifie la variable (\w+(?:\.\w+)*) à (.+)", user_input, re.IGNORECASE)
        for match in matches:
            var_name, var_value = match
            modifications[var_name.strip()] = var_value.strip()
    return modifications

def extract_structured_summary(response):
    start_token = "Résumé Structuré :"
    end_token = "Fin du Résumé"
    start_index = response.find(start_token)
    end_index = response.find(end_token, start_index)
    if start_index != -1 and end_index != -1:
        summary = response[start_index + len(start_token):end_index].strip()
        return summary
    else:
        logging.warning("Le résumé structuré n'a pas pu être extrait.")
        return None

# === AgentManager ===
def agent_manager(chat_history, user_input):
    variables_context = get_variables_context()
    system_prompt = manager_prompt_template.format(variables_context=variables_context)

    conversation = [{"role": "system", "content": system_prompt}]

    # Ajouter l'historique
    for turn in chat_history:
        conversation.append({"role": "user", "content": turn['user']})
        conversation.append({"role": "assistant", "content": turn['assistant']})

    # Ajouter l'entrée utilisateur actuelle
    conversation.append({"role": "user", "content": user_input})

    # Vérifier si l'utilisateur souhaite modifier des variables
    modifications = extract_modifications(user_input)
    if modifications:
        update_project_state(modifications)
        response = "Les variables ont été mises à jour selon votre demande."
        chat_history.append({'user': user_input, 'assistant': response})
        return response, chat_history, False

    # Générer la réponse
    prompt = ""
    for msg in conversation:
        prompt += f"{msg['role']}: {msg['content']}\n"

    input_ids = manager_tokenizer.encode(prompt, return_tensors="pt").to(manager_model.device)
    output_ids = manager_model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=manager_tokenizer.eos_token_id,
        pad_token_id=manager_tokenizer.pad_token_id,
        attention_mask=input_ids.new_ones(input_ids.shape)
    )
    response = manager_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    chat_history.append({'user': user_input, 'assistant': response})

    # Vérifier si un résumé a été généré pour validation
    if "Validez-vous ce résumé" in response:
        structured_summary = extract_structured_summary(response)
        project_state["AgentManager"]["structured_summary"] = structured_summary
        return response, chat_history, True  # Indique que le résumé est prêt pour validation
    else:
        return response, chat_history, False

# --- AgentResearcher ---
# Fonctions spécifiques pour les recherches dynamiques

def fetch_webpage(url: str) -> str:
    """
    Télécharge le contenu HTML d'une URL donnée.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        logging.info(f"Page téléchargée avec succès : {url}")
        return response.text
    except requests.RequestException as e:
        logging.error(f"Erreur lors de la récupération de la page {url}: {e}")
        return ""

def extract_information_from_html(html: str, keyword: str) -> list:
    """
    Extrait des informations pertinentes depuis le HTML en fonction d'un mot-clé.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for code_block in soup.find_all("code"):
            if keyword.lower() in code_block.get_text().lower():
                results.append(code_block.get_text())
        logging.info(f"Nombre de sections extraites pour '{keyword}' : {len(results)}")
        return results
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction des informations : {e}")
        return []

def search_gradio_docs(query: str) -> dict:
    """
    Recherche dans la documentation Gradio les sections pertinentes pour une requête donnée.
    """
    url = "https://gradio.app/docs/"
    logging.info(f"Lancement de la recherche pour la requête : {query}")
    html_content = fetch_webpage(url)
    if not html_content:
        return {"error": "Impossible de télécharger la documentation Gradio."}
    results = extract_information_from_html(html_content, query)
    if not results:
        return {"error": f"Aucun résultat trouvé pour '{query}'."}
    return {"query": query, "results": results}

def agent_researcher():
    structured_summary = project_state["AgentManager"]["structured_summary"]
    if not structured_summary:
        return "Le résumé structuré n'est pas disponible."

    # Création du prompt en utilisant apply_chat_template
    messages = [
        {"role": "system", "content": "Vous êtes un assistant de recherche. Vous devez répondre en JSON avec les clés 'documentation' et 'extraits_code'."},
        {"role": "user", "content": researcher_prompt_template.format(structured_summary=structured_summary)}
    ]

    input_ids = researcher_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(researcher_model.device)

    terminators = [
        researcher_tokenizer.eos_token_id,
        researcher_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    output_ids = researcher_model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response_ids = output_ids[0][input_ids.shape[-1]:]
    response = researcher_tokenizer.decode(response_ids, skip_special_tokens=True)

    # Parser la réponse JSON
    try:
        response_json = json.loads(response)
    except json.JSONDecodeError:
        logging.error("La réponse du modèle n'est pas un JSON valide.")
        response_json = {"documentation": "", "extraits_code": ""}

    # Recherches dynamiques
    search_results = search_gradio_docs(structured_summary)
    if "error" in search_results:
        logging.error(search_results["error"])
        return search_results["error"]

    # Mise à jour de l'état global
    project_state["AgentResearcher"]["search_results"] = {
        "model_response": response_json,
        "dynamic_results": search_results["results"]
    }

    return f"Résultats de l'AgentResearcher :\n{response_json}\n\nRésultats dynamiques :\n{search_results['results']}"

# --- AgentAnalyzer ---
def agent_analyzer():
    structured_summary = project_state["AgentManager"]["structured_summary"]
    search_results = project_state["AgentResearcher"]["search_results"]
    if not structured_summary or not search_results:
        return "Les informations nécessaires ne sont pas disponibles pour l'analyse."

    # Création du prompt avec apply_chat_template
    messages = [
        {"role": "system", "content": "Vous êtes un assistant d'analyse. Votre tâche est d'analyser les résultats de recherche et de vérifier leur cohérence avec le résumé structuré."},
        {"role": "user", "content": analyzer_prompt_template.format(
            structured_summary=structured_summary,
            search_results=json.dumps(search_results, ensure_ascii=False)
        )}
    ]
    prompt = analyzer_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # Création du pipeline
    analyzer_pipeline = transformers.pipeline(
        "text-generation",
        model=analyzer_model,
        tokenizer=analyzer_tokenizer,
        device_map="auto"
    )

    # Génération du rapport d'analyse
    sequences = analyzer_pipeline(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        max_new_tokens=256,
    )
    analysis_report = sequences[0]['generated_text']

    # Mise à jour de l'état global
    project_state["AgentAnalyzer"]["analysis_report"] = analysis_report

    # Détermination de la validité
    if "Validité: Oui" in analysis_report:
        instruction_for_coder = f"Générer du code basé sur :\n{structured_summary}\n\nRésultats de recherche :\n{search_results}"
        project_state["AgentAnalyzer"]["instruction_for_coder"] = instruction_for_coder
        return f"Rapport valide.\nInstructions pour l'AgentCoder prêtes."
    elif "Validité: Non" in analysis_report:
        project_state["AgentAnalyzer"]["instruction_for_coder"] = None
        # Retourner le rapport à l'AgentManager pour clarification
        return f"Rapport non valide. Besoin de clarification.\n{analysis_report}"
    else:
        project_state["AgentAnalyzer"]["instruction_for_coder"] = None
        return f"Le rapport d'analyse ne contient pas d'information claire sur la validité. Besoin de clarification.\n{analysis_report}"

# --- AgentCoder ---
def agent_coder():
    instruction_for_coder = project_state["AgentAnalyzer"]["instruction_for_coder"]
    if not instruction_for_coder:
        return "Les instructions pour le code ne sont pas disponibles."

    # Création des messages avec apply_chat_template
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": instruction_for_coder}
    ]
    prompt = coder_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Préparation des entrées du modèle
    model_inputs = coder_tokenizer(prompt, return_tensors="pt").to(coder_model.device)

    # Génération du code
    generated_ids = coder_model.generate(
        **model_inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )
    # Exclure les tokens du prompt des sorties
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
    final_code = coder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Mise à jour de l'état global
    project_state["AgentCoder"]["final_code"] = final_code

    return f"Code généré par l'AgentCoder :\n{final_code}"

# === Fonction d'interaction avec l'utilisateur ===
def user_interaction(message, chat_history):
    if chat_history is None:
        chat_history = []

    # Vérifier si nous attendons une validation
    if chat_history and isinstance(chat_history[-1], dict) and chat_history[-1].get('status') == 'awaiting_validation':
        # Traiter la validation de l'utilisateur
        user_validation = message
        if user_validation.lower() in ["oui", "yes"]:
            # Procéder avec les agents
            researcher_response = agent_researcher()
            analyzer_response = agent_analyzer()
            if "valide" in analyzer_response.lower():
                coder_response = agent_coder()
                response = coder_response
            else:
                response = analyzer_response
        else:
            response = "Le résumé structuré n'a pas été validé. Veuillez fournir plus de détails."
        # Retirer le statut de chat_history
        chat_history.pop()
        chat_history.append({'user': message, 'assistant': response})
        return chat_history, chat_history
    else:
        # Interaction régulière avec l'AgentManager
        response, chat_history, is_summary_ready = agent_manager(chat_history, message)
        if is_summary_ready:
            # Indiquer que nous attendons une validation
            chat_history.append({'status': 'awaiting_validation'})
        return chat_history, chat_history

# === Interface Gradio ===
with gr.Blocks() as interface:
    chatbot = gr.Chatbot()
    state = gr.State([])
    msg = gr.Textbox(placeholder="Entrez votre message ici...")
    send_btn = gr.Button("Envoyer")

    def respond(message, chat_history):
        updated_chat_history, _ = user_interaction(message, chat_history)
        bot_message = updated_chat_history[-1]['assistant']
        chatbot.append((message, bot_message))
        return chatbot, updated_chat_history

    send_btn.click(respond, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(respond, inputs=[msg, state], outputs=[chatbot, state])

if __name__ == "__main__":
    interface.launch()
