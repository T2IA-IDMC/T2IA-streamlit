from pathlib import Path
import os
from PIL import Image
import pandas as pd
import base64
import io
from openai import OpenAI
import json


GPT_MODEL = 'ft:gpt-4o-2024-08-06:idmc:stampreader-train-150:B9b5lPlH'

system_content = (
    "Tu es un expert en philatélie, spécialisé dans l'analyse des tampons d'oblitération français. "
    "Ton rôle est d'extraire et d'analyser les informations des tampons d'oblitération de courriers du début du 20e siècle, "
    "ou de la fin du 19e, en me retournant les informations uniquement au format JSON structuré. Voici le format exact attendu :\n\n"
    "{\n"
    '    "stamp": "NOM_DU_FICHIER",\n'
    '    "postal agency": "LIEU_DE_LEVÉE" ou null,\n'
    '    "date": "YYYY-MM-DDTHH:mm",\n'
    '    "department": "DÉPARTEMENT" ou null,\n'
    '    "starred hour": true/false,\n'
    '    "collection": "NUMÉRO_DE_LA_LEVÉE" ou null,\n'
    '    "stamp type": "date stamp" ou "hexagonal date stamp" ou "conveyor line stamp",\n'
    '    "quality": "poor", "mediocre", ou "good"\n'
    "}\n\n"
    "### Explications et conventions :\n"
    "- **postal agency** : La ville où le courrier a été levé. Si une partie du texte est illisible ou absente, tu peux essayer de deviner en fonction du département si ce dernier est lisible (Si le département est 'VOSGES', la ville sera une ville des Vosges). Pour les convoyeurs de lignes, on y trouve les noms des deux gares entre lesquels circule le convoyeur (comme 'NANCY A REIMS').\n"
    "- **date** : La date complète au format ISO 8601 court (YYYY-MM-DDTHH:mm). Si une partie (comme les minutes ou l'heure) est manquante ou illisible, remplace-la par 'X' (exemple : 1918-XX-13T21:XX).\n"
    "- **department** : Le département où la levée a eu lieu. Il peut être deviné en fonction du lieu (exemple : si le lieu est 'NANCY', le département sera sans aucun doute 'MTHE-ET-MLLE', abréviation de Meurthe-et-Moselle). Si le tampon est un convoyeur de ligne, ce champ sera null.\n"
    "- **starred hour** : True si une étoile (*) indique une réception après l'heure de levée, sinon False. La date présentera donc toujours des 'X' à la place des minutes dans ce cas.\n"
    "- **collection** : Si le tampon indique un numéro de levée (exemple : '1°', '3E'), spécifie-le ici. Si non spécifié, indique null.\n"
    "- **stamp type** : Trois valeurs possibles : 'date stamp' (tampon rond), 'hexagonal date stamp' (hexagonal), ou 'conveyor line stamp' (la plupart du temps, le tampon est crénelé pour les convoyeurs de ligne, mais il peut être rond).\n"
    "- **quality** : Évalue la qualité globale du tampon :\n"
    "  - 'poor' : Illisible ou presque inutilisable.\n"
    "  - 'mediocre' : Partiellement lisible, mais certaines informations sont incertaines.\n"
    "  - 'good' : Lisible sans ambiguïtés.\n\n"
    "### Contexte supplémentaire :\n"
    "- Les tampons d'oblitération français comportent généralement un lieu sur le pourtour (ex. 'TROYES' au dessus et 'AUBE' en dessous), une date au milieu (ex. '14 -7 13' pour '1913-07-14') avec parfois une heure au dessus de cette date, ou un numéro de levée.\n"
    "- Si le tampon est un convoyeur de ligne, il inclut les noms de deux gares (ex. 'NANCY A LANGRES').\n"
    "- Tu recevras principalement des tampons d'oblitération de courriers du début du 20e siècle, ou de la fin du 19e siècle.\n"
    "- Le tampon peut-être orienté dans n'importe quel sens, il faudra faire attention à l'orientation du texte. Tu peux utiliser la date pour trouver la bonne l'orientation.\n"
    "- Si tu ne comprends pas ou si une information manque dans la date ou l'heure, utilise des 'X' pour signaler l'incertitude.\n"
    "- Fournis toujours une réponse bien formatée, fidèle au format JSON demandé et rien d'autre.\n"
    "- Si le tampon est illisible ou que tu ne vois que le timbre, réponds toujours sous forme d'un json avec 'null' dans les champs correspondants et 'XXXX-XX-XXTXX:XX' pour la date.\n\n"
)



# les images à envoyer à ChaGPT doivent être en base64
def img_array_to_base64(bin_img):
    # Convert the image array to a PIL image
    pil_image = Image.fromarray(bin_img)

    # Convertir l'image en binaire
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return base64_string


def get_GPT_response(base64_image, image_title, system_content, model=GPT_MODEL, api_key=None):
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Nom de l'image : {image_title}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )

    return response


def empty_json(file_name):

    empty_json = (
        "{\n"
        f'    "stamp": "{file_name}",\n'
        '    "postal agency": null,\n'
        '    "date": "XXXX-XX-XXTXX:XX",\n'
        '    "department": null,\n'
        '    "starred hour": false,\n'
        '    "collection": null,\n'
        '    "stamp type": null,\n'
        '    "quality": "poor"\n'
        "}"
    )
    return empty_json


def process_GPT4_response(response, title):
    # on récupère le contenu de la réponse
    content = response.choices[0].message.content

    # on supprime les balises de code JSON
    if isinstance(content, str):
        content = content.replace('```json\n', '').replace('```', '')

    try:
        res = pd.Series(json.loads(content))
    except Exception as e:
        res = pd.Series(json.loads(empty_json(title)))

    return res