from pathlib import Path
from typing import Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from ultralytics import YOLO


class HarnessInspector:
    """Module réutilisable pour l'inspection et l'analyse géométrique par YOLO."""

    def __init__(
        self,
        model_path: Union[str, Path],
        fallback_dir: Union[str, Path, None] = None,
        conf_threshold: float = 0.25,
        img_size: int = 640,
    ):
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.model_path = self._resolve_model_path(Path(model_path), fallback_dir)
        self.model = YOLO(str(self.model_path))
        self.neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    @staticmethod
    def _resolve_model_path(model_path: Path, fallback_dir: Union[str, Path, None]) -> Path:
        if model_path.exists():
            return model_path.resolve()
        if fallback_dir:
            found_weights = sorted(
                Path(fallback_dir).glob("**/best.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if found_weights:
                return found_weights[0].resolve()
        raise FileNotFoundError(f"Modèle YOLO introuvable : {model_path}")

    @staticmethod
    def _cluster_pts(pts_xy: list[tuple[int, int]], min_dist: float = 8.0) -> list[tuple[int, int]]:
        clusters = []
        for x, y in pts_xy:
            merged = False
            for c in clusters:
                cx, cy, n = c
                if np.hypot(x - cx / n, y - cy / n) <= min_dist:
                    c[0] += x
                    c[1] += y
                    c[2] += 1
                    merged = True
                    break
            if not merged:
                clusters.append([float(x), float(y), 1])
        return [(int(round(cx / n)), int(round(cy / n))) for cx, cy, n in clusters]

    def predict(
        self,
        image_source: Union[str, Path, np.ndarray],
        conf: Union[float, None] = None,
        show_plot: bool = False,
        save_vis_path: Union[str, Path, None] = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Exécute la détection/segmentation et extrait les mesures géométriques.
        
        Args:
            image_source: Chemin vers une image ou image OpenCV (array BGR).
            conf: Seuil de confiance optionnel (remplace celui par défaut).
            show_plot: Affiche le résultat avec Matplotlib si True.
            save_vis_path: Chemin pour sauvegarder l'image annotée si renseigné.
            
        Returns:
            (df_resultats, image_annotee_bgr)
        """
        conf_val = conf if conf is not None else self.conf_threshold

        # Chargement de l'image
        if isinstance(image_source, (str, Path)):
            img_path = Path(image_source).resolve()
            if not img_path.exists():
                raise FileNotFoundError(f"Image introuvable : {img_path}")
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise ValueError(f"Impossible de lire l'image : {img_path}")
        elif isinstance(image_source, np.ndarray):
            img_bgr = image_source.copy()
        else:
            raise TypeError("image_source doit être un chemin (str/Path) ou un numpy.ndarray")

        h, w = img_bgr.shape[:2]

        # Inférence YOLO
        res = self.model.predict(
            source=img_bgr, conf=conf_val, imgsz=self.img_size, verbose=False
        )[0]

        results_rows = []
        geometry_overlays = []

        if res.boxes is not None and len(res.boxes) > 0:
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            masks_np = res.masks.data.cpu().numpy() if res.masks is not None else None

            for idx in range(len(cls_ids)):
                cid = int(cls_ids[idx])
                cname = self.model.names.get(cid, str(cid))
                score = round(float(confs[idx]), 3)
                x1, y1, x2, y2 = [round(float(v), 1) for v in boxes_xyxy[idx]]

                length_px = 0
                nb_end = 0
                nb_br = 0

                if masks_np is not None and idx < len(masks_np):
                    mask_u8 = cv2.resize(
                        (masks_np[idx] > 0.5).astype(np.uint8) * 255,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    if int((mask_u8 > 0).sum()) > 0:
                        closed = cv2.morphologyEx(
                            mask_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
                        )
                        skel = skeletonize(closed > 0).astype(np.uint8)
                        coords_yx = np.argwhere(skel > 0)
                        length_px = int(len(coords_yx))

                        if length_px > 0:
                            neigh = cv2.filter2D(
                                skel, ddepth=cv2.CV_16S, kernel=self.neighbor_kernel
                            )
                            active_n = neigh[skel > 0]
                            endpoints = self._cluster_pts(
                                [(int(x), int(y)) for y, x in coords_yx[active_n == 1]], 8.0
                            )
                            branches = self._cluster_pts(
                                [(int(x), int(y)) for y, x in coords_yx[active_n >= 3]], 8.0
                            )
                            nb_end = len(endpoints)
                            nb_br = len(branches)
                            geometry_overlays.append((skel, endpoints, branches))

                results_rows.append({
                    "id": idx,
                    "classe": cname,
                    "confiance": score,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "longueur_px": length_px,
                    "extremites": nb_end,
                    "intersections": nb_br,
                })

        df_resultats = pd.DataFrame(results_rows)

        # Génération de l'image annotée (boîtes/masques YOLO + squelette/points clés)
        annotated_bgr = res.plot()
        for skel, endpoints, branches in geometry_overlays:
            annotated_bgr[skel > 0] = (0, 255, 255)  # Squelette en jaune
            for ex, ey in endpoints:
                cv2.circle(annotated_bgr, (ex, ey), 4, (0, 0, 255), -1)  # Extrémités en rouge
            for bx, by in branches:
                cv2.circle(annotated_bgr, (bx, by), 4, (255, 0, 0), -1)  # Intersections en bleu

        if save_vis_path:
            cv2.imwrite(str(save_vis_path), annotated_bgr)

        if show_plot:
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"Détections : {len(df_resultats)} objet(s)")
            plt.show()

        return df_resultats, annotated_bgr