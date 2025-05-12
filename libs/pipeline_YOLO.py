

# LIBRAIRIES
# ==========
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
from ultralytics import YOLO


# DETECTION
# =========
def yolo_detect(image_path: list[Path], model_path: Path, verbose=False):
    """
    Renvoie le résultat de détection d'une image, ou d'une liste d'images selon le modèle Yolo spécifié.
    """
    model_yolo = YOLO(model_path)
    return model_yolo(image_path, verbose=verbose)


# Ajustement de la taille des bboxes des tampons
# ----------------------------------------------
def adjust_boxes_tensor(boxes, image_shape, margin=0.01):
    """
    Augmente la taille de plusieurs boîtes de 1% par rapport à la taille de l'image.
    """
    height, width = image_shape[0], image_shape[1]

    # Calcul des marges
    margin_width = width * margin
    margin_height = height * margin

    # Ajustement des boîtes
    x1 = torch.clamp(boxes[:, 0] - margin_width, min=0)
    y1 = torch.clamp(boxes[:, 1] - margin_height, min=0)
    x2 = torch.clamp(boxes[:, 2] + margin_width, max=width)
    y2 = torch.clamp(boxes[:, 3] + margin_height, max=height)

    # Combinaison des résultats
    adjusted_boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # Conversion en entiers arrondis
    return torch.round(adjusted_boxes).to(dtype=torch.int32)


# Processing des résultats de Yolo pour Yolo-seg
# ----------------------------------------------
def yolo2yolo_seg(detection):
    """
    Transforme les résultats de détections de Yolo, uniquement pour la classe spécifiée, pour segmentation.
    Retourne une liste d'images rognées et les boîtes englobantes ajustées correspondantes

    Utilise la fonction adjust_boxes_tensor() pour faciliter la segmentation.
    """
    # liste des images des tampons
    cropped_list = []

    # on récupère l'image d'origine en BGR et convertie en RGB
    image = detection.orig_img[:, :, ::-1]

    # on ajuste les boîtes englobantes
    adjusted_boxes = adjust_boxes_tensor(detection.boxes.xyxy, image.shape)

    ## tester si des tampons ont été détectés, sinon on ne fait rien
    for box in adjusted_boxes:
        x_min, y_min, x_max, y_max = box
        cropped_list.append(image[y_min:y_max, x_min:x_max])

    return cropped_list, adjusted_boxes


# SEGMENTATION
# ============
def yolo_seg(cropped_img, model_seg_path, verbose=False):
    """
    Renvoie le résultat de segmentation d'une image selon le modèle Yolo spécifié
    """
    model_yolo = YOLO(model_seg_path)
    return model_yolo(cropped_img, verbose=verbose)


# Affichage des résultats de segmentation
# ---------------------------------------
def plot_seg_mask(segmentation, orig_img=True, ax=None):
    if ax is None:
        ax = plt.gca()

    # masque
    mask = segmentation.masks.data.cpu().numpy().sum(axis=0) > 0
    if orig_img:
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape
        mask = mask.astype(np.uint8)
        mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        # affichage
        image = Image.fromarray(segmentation.orig_img.copy())
        image = np.array(image.resize((w, h)))
        ax.imshow(image)
        ax.imshow(mask)
    else:
        ax.imshow(mask, cmap='gray')


# Binarisation du masque
# ----------------------
def get_bin_stamp(segmentation):
    return segmentation.masks.data.cpu().numpy().sum(axis=0) == 0


# PIPELINE
# ========
def pipeline_yolo2(image_path: list[Path], model_path: Path, model_seg_path: Path, cls='official postmark', verbose=False):

    results = {'img_name': [], 'path':[], 'boxes':[], 'boxes_adj':[],
               'boxes_cls': [], 'stamps_bins':[], 'stamps_titles':[]}

    detections = yolo_detect(image_path, model_path, verbose=verbose)

    for detection in detections:
        # on récupère la clef de la classe voulue
        postmark_key = [key for (key, val) in detection.names.items() if val == cls][0]

        # résultats de détection
        img_path = Path(detection.path)
        results['path'].append(img_path)
        results['img_name'].append(img_path.stem)
        boxes = detection.boxes.xyxy.clone().detach().round().type(torch.int)
        results['boxes'].append(boxes.cpu().numpy())
        boxes_cls = detection.boxes.cls.clone().detach().type(torch.int)
        results['boxes_cls'].append(boxes_cls.cpu().numpy())

        cropped_imgs, adjusted_boxes = yolo2yolo_seg(detection)

        results['boxes_adj'].append(adjusted_boxes.cpu().numpy())

        # segmentation
        stamps_bins = np.empty((len(boxes_cls),), dtype=object)
        stamp_titles = np.empty((len(boxes_cls),), dtype=object)
        for i, (cropped_img, cls_nb) in enumerate(zip(cropped_imgs, boxes_cls)):

            if cls_nb == postmark_key:  # segmentation des tampons
                segmentations = yolo_seg(cropped_img, model_seg_path, verbose=verbose)

                for j, segmentation in enumerate(segmentations):
                    if segmentation.masks is not None:  # si des segmentations ont été trouvées
                        stamps_bins[i] = get_bin_stamp(segmentation)

                    else:
                        stamps_bins[i] = np.ones(cropped_img.shape[:-1], dtype=bool)

                    stamp_titles[i] = f'{img_path.stem}_{j}'

            else:
                stamps_bins[i] = np.nan  # ou cropped_img ?
                stamp_titles[i] = np.nan

        results['stamps_bins'].append(stamps_bins)
        results['stamps_titles'].append(stamp_titles)

    return pd.DataFrame(results)

