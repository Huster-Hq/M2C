import os
import shutil
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import subprocess
import csv 
from PIL import Image
from tqdm import tqdm
import glob
from collections import defaultdict
import random
import argparse

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.geometry_encoders import Prompt
from sam3.model.box_ops import box_xyxy_to_cxcywh

WORKSPACE_DIR  = "M2C-main/datasets/deac_workspace"
LATEST_CKPT_LINK = os.path.join(WORKSPACE_DIR, "phi_continuous_checkpoint.pt")
DATA_LOG_PATH    = os.path.join(WORKSPACE_DIR, "tta_analysis_data.csv")
GLOBAL_EVAL_LOG = os.path.join(WORKSPACE_DIR, "global_sequential_eval.csv")

CURVE_LOG_PATH = os.path.join(WORKSPACE_DIR, "strategy_learning_curves.csv")
TRAIN_SCRIPT   = "M2C-main/sam3/train/train.py" 
BASE_CONFIG    = "M2C-main/sam3/train/configs/kvasir-seg/prompt.yaml"

BATCH_SIZE = 5  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SOLVED_THRESH = 0.6 
TEXT_PROMPT = "A polyp in a colonoscopy image"


def yolo_to_bbox(x_center, y_center, w, h, img_width, img_height):
    w_pixel = w * img_width
    h_pixel = h * img_height
    x_min = int((x_center * img_width) - (w_pixel / 2))
    y_min = int((y_center * img_height) - (h_pixel / 2))
    return x_min, y_min, int(w_pixel), int(h_pixel)

def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0: return 1.0 if (mask1.sum()+mask2.sum())==0 else 0.0
    return inter / (union + 1e-6)


def compute_dice(pred, gt):
    if gt is None: return 0.0
    p = (pred > 0).astype(np.float32)
    g = (gt > 0).astype(np.float32)
    inter = (p * g).sum()
    total = p.sum() + g.sum()
    if total == 0: return 1.0
    return 2 * inter / total


class DEAC_Controller:
    def __init__(self):
        if os.path.exists(WORKSPACE_DIR): shutil.rmtree(WORKSPACE_DIR)
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        
        self.csv_log_path = DATA_LOG_PATH
        with open(self.csv_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Image',  'Entropy', 'IoU_Cycle', 'Unc_Score', 'Real_Dice'])

        with open(GLOBAL_EVAL_LOG, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Labeled_Count', 'Global_Dice', 'Global_Dice_Std', 'Global_IoU', 'Global_IoU_Std', 'Auto_Solved_Count'])
        
        print(">>> Loading SAM3 Model...")
        self.model = build_sam3_image_model().to(DEVICE)
        self.processor = Sam3Processor(self.model)
        self.model.eval()
        
        self.text_encoder = self.model.backbone.language_backbone
        token_ids = self.text_encoder.tokenizer([TEXT_PROMPT], context_length=self.text_encoder.context_length).to(DEVICE)
        with torch.no_grad():
            self.base_text_embeds = self.text_encoder.encoder.token_embedding(token_ids)

        self.all_images = sorted([f for f in os.listdir(POOL_IMG_DIR) if f.lower().endswith(('.jpg','.png'))])
        self.uncovered_pool = self.all_images.copy() 
        self.round_idx = 0

        self.labeled_files = set()
        self.expert_library = [] 
        self.global_inference_cache = defaultdict(dict)
        self.gt_cache = {}
        self.full_map_cache = {} 
        self._cache_pool_features()
        self.test_pred_cache = {}

    def _extract_feats(self, pil_img, mask_path=None):
        with torch.no_grad():
            inf = self.processor.set_image(pil_img)
            feats = inf["backbone_out"]["vision_features"]
            if isinstance(feats, list): feats = feats[-1]
            feats = F.avg_pool2d(feats, kernel_size=3, stride=1, padding=1)
            feats = F.normalize(feats.squeeze(0), p=2, dim=0) 
            
            if mask_path and os.path.exists(mask_path):
                mask_cv = cv2.imread(mask_path, 0)
                mask_res = cv2.resize(mask_cv, (feats.shape[2], feats.shape[1]), interpolation=cv2.INTER_NEAREST)
                fg_idx = np.where(mask_res > 0)
                if len(fg_idx[0]) > 0:
                    return feats[:, fg_idx[0], fg_idx[1]].permute(1, 0)
                return None
            return feats.cpu()

    def _cache_pool_features(self):
        print(">>> [Init] Caching features for all pool images...")
        for f in tqdm(self.all_images):
            path = os.path.join(POOL_IMG_DIR, f)
            pil = Image.open(path).convert("RGB")
            self.full_map_cache[f] = self._extract_feats(pil)

    def compute_advanced_score(self, test_feat_map, library_feats):
        if library_feats is None: return 0.0, None
        test_feat_map = test_feat_map.to(DEVICE)
        library_feats = library_feats.to(DEVICE)
        C, H, W = test_feat_map.shape
        test_feat_flat = test_feat_map.view(C, -1).permute(1, 0)
        sim_matrix = torch.mm(test_feat_flat, library_feats.t())
        pixel_scores, _ = torch.max(sim_matrix, dim=1) 
        pixel_scores = pixel_scores.view(1, 1, H, W)
        peak_map = F.max_pool2d(pixel_scores, kernel_size=5, stride=1, padding=2)
        k = max(1, int(H * W * 0.02))
        top_v, _ = torch.topk(peak_map.view(-1), k=k)
        peak_score = torch.mean(top_v).item()
        global_mean = torch.mean(pixel_scores).item()
        snr = peak_score / (global_mean + 1e-6)
        return peak_score * snr, pixel_scores.detach().cpu()

    def select_batch(self, current_pool, uncertainty_map=None):
        if len(current_pool) <= BATCH_SIZE:
            return current_pool
            
        if self.round_idx == 1:
            print(f"  [Strategy] Round 1: Random Selection")
            batch = random.sample(current_pool, n_shot) 
            return batch
        else:
            print(f"  [Strategy] Round {self.round_idx}: Pure Top-{BATCH_SIZE} Hardest Samples")
            if not uncertainty_map:
                return current_pool[:BATCH_SIZE]
            
            pool_scores = []
            for f in current_pool:
                score = uncertainty_map.get(f, -1.0)
                pool_scores.append((f, score))
            
            pool_scores.sort(key=lambda x: x[1], reverse=True)
            top_k = pool_scores[:BATCH_SIZE]
            batch = [x[0] for x in top_k]
            
            print(f"  > Selected Top-{BATCH_SIZE} Hardest:")
            for i, (fname, score) in enumerate(top_k):
                print(f"    {i+1}. {fname} (Unc: {score:.4f})")
                
            return batch

    def run_training(self, selected_files):
        round_name = f"round_{self.round_idx}"
        round_dir = os.path.join(WORKSPACE_DIR, round_name)
        train_img = os.path.join(round_dir, "images"); os.makedirs(train_img, exist_ok=True)
        train_mask = os.path.join(round_dir, "masks"); os.makedirs(train_mask, exist_ok=True)
        train_lbl = os.path.join(round_dir, "labels"); os.makedirs(train_lbl, exist_ok=True)
        
        valid_files = []
        for fname in selected_files:
            stem = os.path.splitext(fname)[0]
            shutil.copy(os.path.join(POOL_IMG_DIR, fname), os.path.join(train_img, fname))
            shutil.copy(os.path.join(POOL_MASK_DIR, stem + ".png"), os.path.join(train_mask, stem + ".png"))
            valid_files.append(fname)
        
        json_path = os.path.join(round_dir, "train.json")
        self.generate_coco_json(valid_files, train_img, train_mask, train_lbl, json_path)
        
        cmd = ["python", TRAIN_SCRIPT, "-c", BASE_CONFIG, "--override_data", 
               "--train_img_dir", train_img, "--train_json_path", json_path, 
               "--output_path", round_dir, "--epochs", "60", 
               "--use-cluster", "0", "--batch_size", str(len(valid_files))]
        
        with open(os.path.join(round_dir, "train.log"), "w") as logf: 
            subprocess.check_call(cmd, stdout=logf, stderr=subprocess.STDOUT)
        
        ckpt = os.path.join(round_dir, "phi_continuous_checkpoint.pt")
        if os.path.exists(ckpt):
            shutil.copy(ckpt, LATEST_CKPT_LINK) 
            return ckpt
        return None

    def _predict_mask_raw(self, pil_img, text_outputs):
        inference_state = self.processor.set_image(pil_img)
        inference_state["backbone_out"].update(text_outputs)
        inference_state["geometric_prompt"] = Prompt(
            box_embeddings=torch.zeros(0, 1, 4, device=DEVICE), box_mask=torch.zeros(1, 0, device=DEVICE, dtype=torch.bool),
            point_embeddings=torch.zeros(0, 1, 2, device=DEVICE), point_mask=torch.zeros(1, 0, device=DEVICE, dtype=torch.bool),
            box_labels=torch.zeros(0, 1, device=DEVICE, dtype=torch.long), point_labels=torch.zeros(0, 1, device=DEVICE, dtype=torch.long),
        )
        out = self.processor._forward_grounding(inference_state)
        w, h = pil_img.size
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        if "masks" in out and len(out["masks"]) > 0:
            for i, mask_tensor in enumerate(out["masks"]):
                if mask_tensor.dim() > 2:
                    mask_tensor = mask_tensor.squeeze(0)
                mask_np = mask_tensor.detach().cpu().numpy()
                binary_mask = (mask_np > 0).astype(np.uint8) * 255
                final_mask = np.maximum(final_mask, binary_mask)
        return final_mask, out

    def _predict_mask_with_pred_boxes(self, pil_img, pred_boxes_abs):

        w, h = pil_img.size
        device = DEVICE
        
        inference_state = self.processor.set_image(pil_img)
        scale = torch.tensor([w, h, w, h], device=device, dtype=torch.float32)
        boxes_norm = pred_boxes_abs / scale
        boxes_cxcywh = box_xyxy_to_cxcywh(boxes_norm)
        
        num_boxes = boxes_cxcywh.shape[0]
        
        # [N, 4] -> [N, 1, 4]
        box_tensor = boxes_cxcywh.unsqueeze(1) 
        
        # [N, 1] 
        box_labels = torch.ones(num_boxes, 1, device=device, dtype=torch.long)
        
        self.processor.reset_all_prompts(inference_state)
        
        if "geometric_prompt" not in inference_state:
            inference_state["geometric_prompt"] = self.model._get_dummy_prompt()
            
        inference_state["geometric_prompt"].append_boxes(box_tensor, box_labels)
        
        dummy_text = self.model.backbone.forward_text(captions=["visual"], device=device)
        inference_state["backbone_out"].update(dummy_text)
        
        out_state = self.processor._forward_grounding(inference_state)
        
        prob_map_tensor = out_state["masks_logits"]
        if prob_map_tensor.numel() == 0:
            return np.zeros((h, w), dtype=np.uint8)
            
        if prob_map_tensor.dim() == 4:
            # [N, 1, H, W] -> [H, W] (Max/Union)
            final_prob_map, _ = torch.max(prob_map_tensor, dim=0)
            final_prob_map = final_prob_map.squeeze(0)
        else:
            final_prob_map = prob_map_tensor

        prob_map = final_prob_map.detach().cpu().numpy()
        
        if prob_map.shape != (h, w):
            prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
            
        return (prob_map > 0.5).astype(np.uint8) * 255
    
    def infer_current_round_on_pool(self, phi_path, pool_files):
        print(f"  [Inference] Force running Round {self.round_idx} on {len(pool_files)} pool images...")
        
        phi_tensor = torch.load(phi_path, map_location=DEVICE)
        if isinstance(phi_tensor, dict): phi_tensor = phi_tensor.get("pt_phi", next(iter(phi_tensor.values())))
        with torch.no_grad():
            v_new = self.base_text_embeds + phi_tensor.to(DEVICE)
            text_out = self.model.backbone.forward_text(captions=[TEXT_PROMPT], input_embeds=v_new, device=DEVICE)

        for fname in tqdm(pool_files, desc=f"Infer R{self.round_idx}", leave=False):
            if self.round_idx in self.global_inference_cache[fname]:
                continue
            
            img_p = os.path.join(POOL_IMG_DIR, fname)
            pil = Image.open(img_p).convert("RGB")
            
            m_pred, out = self._predict_mask_raw(pil, text_out)
            unc_score, iou_c = self._compute_single_image_tta(pil, m_pred, out, text_out)
            
            self.global_inference_cache[fname][self.round_idx] = (m_pred, unc_score, iou_c)

    def filter_pool_by_tta(self, phi_path, current_pool):
        print(f"  [Strategy] Filtering {len(current_pool)} samples (Reading Cache)...")

        hard_samples = []
        easy_samples = []
        uncertainty_map = {}
        pool_dice_accumulator = []
        pool_unc_accumulator = []
        
        easy_dice_accumulator = []
        hard_unc_accumulator = []
        temp_dice_cache = {}

        f_csv = open(self.csv_log_path, 'a', newline='')
        writer = csv.writer(f_csv)

        for fname in tqdm(current_pool, desc="Expert Selection", leave=False):
            if self.round_idx in self.global_inference_cache[fname]:
                m_text, unc_score, iou_c = self.global_inference_cache[fname][self.round_idx]
            else:
                print(f"Error: {fname} not found in cache for R{self.round_idx}!")
                continue

            gt_bin = None
            if fname in self.gt_cache:
                gt_bin = self.gt_cache[fname]
            else:
                gt_p = os.path.join(POOL_MASK_DIR, os.path.splitext(fname)[0] + ".png")
                if os.path.exists(gt_p):
                    gt_mask = cv2.imread(gt_p, 0)
                    h, w = m_text.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    _, gt_bin = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
                    self.gt_cache[fname] = gt_bin
                else:
                    self.gt_cache[fname] = None
            
            real_dice = compute_dice(m_text, gt_bin) if gt_bin is not None else 0.0
            
            uncertainty_map[fname] = unc_score
            temp_dice_cache[fname] = real_dice
                
            ent = max(0.0, unc_score - (1-iou_c))
            
            uncertainty_map[fname] = float(unc_score)

            writer.writerow([self.round_idx, fname,  ent, iou_c, unc_score, real_dice])

            if real_dice != -1.0: pool_dice_accumulator.append(real_dice)
            pool_unc_accumulator.append(unc_score)


            if unc_score < SOLVED_THRESH and iou_c > 0.90:
                easy_samples.append(fname)
                easy_dice_accumulator.append(real_dice)
            else:
                hard_samples.append(fname)
                hard_unc_accumulator.append(unc_score)

        f_csv.close()
        
        return easy_samples, hard_samples, uncertainty_map, temp_dice_cache

    def generate_coco_json(self, image_files, img_dir, mask_dir, label_dir, output_json_path):
        coco_output = {
            "images": [], 
            "annotations": [], 
            "categories": [{"id": 1, "name": "Polyp"}]
        }
        img_id, ann_id = 1, 1
        
        valid_count = 0
        for fname in image_files:
            mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")
            img_p = os.path.join(img_dir, fname)
            
            if not os.path.exists(img_p) or not os.path.exists(mask_path):
                continue
                
            with Image.open(img_p) as pil_img:
                w, h = pil_img.size

            mask_cv = cv2.imread(mask_path, 0)
            if mask_cv is None: continue
            if mask_cv.max() <= 1: mask_cv = mask_cv * 255
            _, binary_mask = cv2.threshold(mask_cv, 127, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            has_ann = False
            for c in contours:
                if cv2.contourArea(c) < 30: continue
                
                x, y, bw, bh = cv2.boundingRect(c)
                
                coco_output["annotations"].append({
                    "id": ann_id, 
                    "image_id": img_id, 
                    "category_id": 1, 
                    "bbox": [x, y, bw, bh], 
                    "segmentation": [c.flatten().tolist()], 
                    "area": float(cv2.contourArea(c)), 
                    "iscrowd": 0
                })
                ann_id += 1
                has_ann = True
            
            if has_ann:
                coco_output["images"].append({
                    "id": img_id, 
                    "file_name": fname, 
                    "width": w, 
                    "height": h
                })
                img_id += 1
                valid_count += 1
            
        with open(output_json_path, 'w') as f: 
            json.dump(coco_output, f)
        
        self.check_annotation_quality(output_json_path, img_dir)
        

    def check_annotation_quality(self, json_path, img_dir):
        save_dir = os.path.join(os.path.dirname(json_path), "vis_debug_labels")
        if os.path.exists(save_dir): shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        img_to_anns = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns: img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
            
        for img_info in data['images']:
            img_id = img_info['id']
            fname = img_info['file_name']
            
            img_p = os.path.join(img_dir, fname)
            img = cv2.imread(img_p)
            if img is None: continue
            
            anns = img_to_anns.get(img_id, [])
            
            for ann in anns:
                x, y, w, h = map(int, ann['bbox'])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(img, [poly], True, (0, 0, 255), 2)
            
            cv2.imwrite(os.path.join(save_dir, f"vis_{fname}"), img)

    def save_expert_memory(self, selected_files, phi_path):
        round_dir = os.path.dirname(phi_path)
        all_fg_feats = []
        for fname in selected_files:
            mask_path = os.path.join(POOL_MASK_DIR, os.path.splitext(fname)[0] + ".png")
            pil = Image.open(os.path.join(POOL_IMG_DIR, fname)).convert("RGB")
            feats = self._extract_feats(pil, mask_path) 
            if feats is not None:
                all_fg_feats.append(feats.cpu())
        if all_fg_feats:
            combined = torch.cat(all_fg_feats, dim=0)
            if combined.shape[0] > 1500: 
                combined = combined[torch.randperm(combined.shape[0])[:1500]]
            save_path = os.path.join(round_dir, "expert_memory.pt")
            torch.save({
                "phi_path": phi_path,
                "fg_features": combined,
                "round": self.round_idx
            }, save_path)
            print(f"  > Memory saved: {save_path}")

    
    def save_round_detail_log(self, round_idx, labeled_files, easy_list, hard_list, unc_map, dice_cache):
        round_dir = os.path.join(WORKSPACE_DIR, f"round_{round_idx}")
        os.makedirs(round_dir, exist_ok=True)
        log_path = os.path.join(round_dir, f"round_{round_idx}_details.csv")
        
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['FileName', 'Status', 'Uncertainty', 'Real_Dice', 'Is_Satisfied(>0.9)'])
            
            for f in labeled_files:
                writer.writerow([f, 'Labeled_Intervention', 0.0, 1.0, 'Yes'])
                
            for f in easy_list:
                u = unc_map.get(f, -1.0)
                d = dice_cache.get(f, 0.0)
                sat = 'Yes' if d > 0.9 else 'No'
                writer.writerow([f, 'Easy_Auto', u, d, sat])
                
            for f in hard_list:
                u = unc_map.get(f, -1.0)
                d = dice_cache.get(f, 0.0) 
                sat = 'Yes' if d > 0.9 else 'No'
                writer.writerow([f, 'Hard_Pending', u, d, sat])
                
    
    def _compute_single_image_tta(self, pil_img, m_text, out, text_out):

        if m_text.sum() == 0:
            return 2.0, 0.0, 0.0 
        
        if out["boxes"].shape[0] > 0:
            m_box = self._predict_mask_with_pred_boxes(pil_img, out["boxes"])
            iou_c = compute_iou(m_text, m_box)
        else:
            iou_c = 0.0
        
        logits = out["masks_logits"]
        if logits.dim() == 4: logits, _ = torch.max(logits, dim=0)
        p_map = logits.squeeze().detach().cpu().numpy()
        p_norm = np.clip(p_map, 1e-6, 1-1e-6)
        uncertain_mask = (p_map > 0.05) & (p_map < 0.95)
        
        if uncertain_mask.sum() > 0:
            pixel_entropy = -(p_norm * np.log(p_norm) + (1-p_norm) * np.log(1-p_norm))
            ent = np.mean(pixel_entropy[uncertain_mask])
        else:
            ent = 0.0
            
        unc_score = (1 - iou_c) + ent
        return float(unc_score), float(iou_c)
    
    def update_expert_library(self, phi_path):
        phi_tensor = torch.load(phi_path, map_location=DEVICE)
        if isinstance(phi_tensor, dict): 
            phi_tensor = phi_tensor.get("pt_phi", next(iter(phi_tensor.values())))
        
        with torch.no_grad():
            v_new = self.base_text_embeds + phi_tensor.to(DEVICE)
            text_out = self.model.backbone.forward_text(captions=[TEXT_PROMPT], input_embeds=v_new, device=DEVICE)
            
        self.expert_library.append((self.round_idx, text_out))

    def evaluate_global_sequential(self):
        
        dice_scores = []
        iou_scores = []
        
        solved_by_auto = 0
        solved_by_human = len(self.labeled_files)
        
        for fname in tqdm(self.all_images, desc=f"Seq Eval (R{self.round_idx})", leave=False):
            
            current_dice = 0.0
            current_iou = 0.0

            if fname in self.labeled_files:
                current_dice = 1.0
                current_iou = 1.0
                
            else:
                final_mask = None
                min_unc_score = 100.0
                best_fallback_mask = None
                pil = None 
                
                for r_id, text_out in self.expert_library:
                    if r_id in self.global_inference_cache[fname]:
                        m_pred, unc_score, iou_c = self.global_inference_cache[fname][r_id]
                    else:
                        if pil is None: 
                            img_p = os.path.join(POOL_IMG_DIR, fname)
                            pil = Image.open(img_p).convert("RGB")
                        m_pred, out = self._predict_mask_raw(pil, text_out)
                        unc_score, iou_c = self._compute_single_image_tta(pil, m_pred, out, text_out)
                        self.global_inference_cache[fname][r_id] = (m_pred, unc_score, iou_c)
                    
                    if unc_score < min_unc_score:
                        min_unc_score = unc_score
                        best_fallback_mask = m_pred
                    
                    is_easy = (unc_score < SOLVED_THRESH) and (iou_c > 0.90)
                    if is_easy:
                        final_mask = m_pred
                        solved_by_auto += 1
                        break 
                
                if final_mask is None:
                    final_mask = best_fallback_mask
                
                gt_bin = None
            
                if fname in self.gt_cache:
                    gt_bin = self.gt_cache[fname]
                else:
                    gt_p = os.path.join(POOL_MASK_DIR, os.path.splitext(fname)[0] + ".png")
                    
                    if os.path.exists(gt_p):
                        gt_mask = cv2.imread(gt_p, 0)
                        
                        if final_mask is not None:
                            h, w = final_mask.shape
                            gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        _, gt_bin_processed = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

                        self.gt_cache[fname] = gt_bin_processed
                        gt_bin = gt_bin_processed
                    else:
                        self.gt_cache[fname] = None
                if gt_bin is not None and final_mask is not None:
                    current_dice = compute_dice(final_mask, gt_bin)
                    current_iou = compute_iou(final_mask, gt_bin)
                else:
                    current_dice = 0.0
                    current_iou = 0.0

            dice_scores.append(current_dice)
            iou_scores.append(current_iou)

        avg_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        avg_iou = np.mean(iou_scores)
        std_iou = np.std(iou_scores)
        
        print(f"  [Global Eval Result] Dice: {avg_dice:.4f} ± {std_dice:.4f} | IoU: {avg_iou:.4f} ± {std_iou:.4f}")
        
        with open(GLOBAL_EVAL_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.round_idx, solved_by_human, f"{avg_dice:.4f}", f"{std_dice:.4f}", f"{avg_iou:.4f}", f"{std_iou:.4f}", solved_by_auto])

    def run_loop(self):
        print(f"Total Pool Size: {len(self.uncovered_pool)}")
        uncertainty_map = None 

        while len(self.uncovered_pool) > 0:
            self.round_idx += 1
            print(f"\nRound {self.round_idx} | Pool: {len(self.uncovered_pool)}")
            
            if len(self.uncovered_pool) <= BATCH_SIZE:
                
                batch_files = list(self.uncovered_pool)
                
                self.labeled_files.update(batch_files)
                phi_path = self.run_training(batch_files)
                
                if not phi_path:
                    print(" Training crashed in End Game."); break
                
                self.update_expert_library(phi_path)
                self.evaluate_global_sequential()

                self.uncovered_pool = []
                break
            
            if few_shot:
                if self.round_idx == 1:
                    batch_files = self.select_batch(self.uncovered_pool, uncertainty_map)
                else:
                    break
            else:
                batch_files = self.select_batch(self.uncovered_pool, uncertainty_map)

            print(f"  [Targets] {batch_files}")
            self.labeled_files.update(batch_files)
            
            phi_path = self.run_training(batch_files)
            if not phi_path:
                print("Training crashed."); break
            
            self.update_expert_library(phi_path)
            if len(self.uncovered_pool) > 0:
                self.infer_current_round_on_pool(phi_path, self.uncovered_pool)
            self.evaluate_global_sequential()

            for f in batch_files: 
                if f in self.uncovered_pool: self.uncovered_pool.remove(f)
            
            if len(self.uncovered_pool) > 0:
                easy_list, hard_list, new_unc_map, round_dice_cache = self.filter_pool_by_tta(phi_path, self.uncovered_pool)
                self.save_round_detail_log(self.round_idx, batch_files, easy_list, hard_list, new_unc_map, round_dice_cache)
                
                print(f"  [Result] Easy (Removed): {len(easy_list)} | Hard (Kept): {len(hard_list)}")
                
                for f in easy_list:
                    if f in self.uncovered_pool: self.uncovered_pool.remove(f)
                
                uncertainty_map = {k: v for k, v in new_unc_map.items() if k in self.uncovered_pool}
            else:
                print("Pool is empty. All samples covered.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DEAC Controller")
    
    parser.add_argument("--pool_root", type=str, default="datasets/Kavsir-seg", 
                        help="Root directory for pool dataset")
    parser.add_argument("--n_shot", type=int, default=5, 
                        help="Number of shots for selection")
    parser.add_argument("--few_shot", action="store_true", 
                        help="Flag to enable few_shot mode")

    args = parser.parse_args()

    POOL_ROOT = args.pool_root
    POOL_IMG_DIR = os.path.join(POOL_ROOT, "images")
    POOL_MASK_DIR = os.path.join(POOL_ROOT, "masks")
    n_shot = args.n_shot
    few_shot = args.few_shot

    controller = DEAC_Controller()
    controller.run_loop()