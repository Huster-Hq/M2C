import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import csv
import glob
from PIL import Image
from tqdm import tqdm
from collections import defaultdict 
import argparse

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.geometry_encoders import Prompt
from sam3.model.box_ops import box_xyxy_to_cxcywh

WORKSPACE_DIR = "M2C-main/datasets/deac_workspace"
TEST_IMG_DIR = "M2C-main/datasets/Kvasir-seg/query/image"
TEST_GT_DIR  = "M2C-main/datasets/Kvasir-seg/query/mask"

VIS_ROOT = os.path.join(WORKSPACE_DIR, "sequential_test_vis")
CURVE_CSV = os.path.join(WORKSPACE_DIR, "sequential_incremental_curve.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_PROMPT = "A polyp in a colonoscopy image"

SOLVED_THRESH = 0.6

def compute_dice(pred, gt):
    if gt is None: return 0.0
    p = (pred > 0).astype(np.float32)
    g = (gt > 0).astype(np.float32)
    inter = (p * g).sum()
    total = p.sum() + g.sum()
    if total == 0: return 1.0
    return 2 * inter / total

def compute_iou_fg(mask1, mask2):
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0: return 1.0 if (mask1.sum()+mask2.sum())==0 else 0.0
    return inter / (union + 1e-6)

class IncrementalSequentialTester:
    def __init__(self):
        if os.path.exists(VIS_ROOT): shutil.rmtree(VIS_ROOT)
        os.makedirs(VIS_ROOT, exist_ok=True)
        
        print(">>> [Init] Loading SAM3 Model...")
        self.model = build_sam3_image_model().to(DEVICE)
        self.processor = Sam3Processor(self.model)
        self.model.eval()
        
        self.text_encoder = self.model.backbone.language_backbone
        token_ids = self.text_encoder.tokenizer([TEXT_PROMPT], context_length=self.text_encoder.context_length).to(DEVICE)
        with torch.no_grad():
            self.base_text_embeds = self.text_encoder.encoder.token_embedding(token_ids)
            
        self.phi_library = [] 
        self._load_phi_library()

    def _load_phi_library(self):
        print(f">>> [Init] Scanning workspace: {WORKSPACE_DIR}")
        round_dirs = sorted(glob.glob(os.path.join(WORKSPACE_DIR, "round_*")), 
                            key=lambda x: int(x.split('_')[-1]))

        for r_dir in round_dirs:
            r_id = int(r_dir.split('_')[-1])
            ckpt_path = os.path.join(r_dir, "phi_continuous_checkpoint.pt")
            if os.path.exists(ckpt_path):
                phi_tensor = torch.load(ckpt_path, map_location=DEVICE)
                if isinstance(phi_tensor, dict): 
                    phi_tensor = phi_tensor.get("pt_phi", next(iter(phi_tensor.values())))
                
                with torch.no_grad():
                    v_new = self.base_text_embeds + phi_tensor.to(DEVICE)
                    text_out = self.model.backbone.forward_text(captions=[TEXT_PROMPT], input_embeds=v_new, device=DEVICE)
                
                self.phi_library.append((r_id, text_out))

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
                if mask_tensor.dim() > 2: mask_tensor = mask_tensor.squeeze(0)
                mask_np = mask_tensor.detach().cpu().numpy()
                binary_mask = (mask_np > 0).astype(np.uint8) * 255
                final_mask = np.maximum(final_mask, binary_mask)
        return final_mask, out 

    def _predict_mask_with_pred_boxes(self, pil_img, pred_boxes_abs):
        w, h = pil_img.size
        inference_state = self.processor.set_image(pil_img)
        scale = torch.tensor([w, h, w, h], device=DEVICE, dtype=torch.float32)
        boxes_norm = pred_boxes_abs / scale
        boxes_cxcywh = box_xyxy_to_cxcywh(boxes_norm)
        num_boxes = boxes_cxcywh.shape[0]
        box_tensor = boxes_cxcywh.unsqueeze(1) 
        box_labels = torch.ones(num_boxes, 1, device=DEVICE, dtype=torch.long)
        self.processor.reset_all_prompts(inference_state)
        if "geometric_prompt" not in inference_state:
            inference_state["geometric_prompt"] = self.model._get_dummy_prompt()
        inference_state["geometric_prompt"].append_boxes(box_tensor, box_labels)
        dummy_text = self.model.backbone.forward_text(captions=["visual"], device=DEVICE)
        inference_state["backbone_out"].update(dummy_text)
        out_state = self.processor._forward_grounding(inference_state)
        prob_map_tensor = out_state["masks_logits"]
        if prob_map_tensor.numel() == 0: return np.zeros((h, w), dtype=np.uint8)
        if prob_map_tensor.dim() == 4:
            final_prob_map, _ = torch.max(prob_map_tensor, dim=0)
            final_prob_map = final_prob_map.squeeze(0)
        else: final_prob_map = prob_map_tensor
        prob_map = final_prob_map.detach().cpu().numpy()
        if prob_map.shape != (h, w): prob_map = cv2.resize(prob_map, (w, h))
        return (prob_map > 0.5).astype(np.uint8) * 255
    
    def visualize_result(self, fname, pil_img, pred_mask, save_name):
        save_dir = os.path.join(VIS_ROOT, os.path.splitext(fname)[0])
        os.makedirs(save_dir, exist_ok=True)
        
        if pred_mask is None:
            w, h = pil_img.size
            save_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            save_mask = pred_mask.astype(np.uint8)
            if save_mask.max() == 1:
                save_mask = save_mask * 255

        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, save_mask)

    def run_incremental_inference(self):
        test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg','.png'))])
        inference_cache = defaultdict(dict)
        
        with open(CURVE_CSV, 'w', newline='') as f_curve:
            writer_curve = csv.writer(f_curve)
            writer_curve.writerow(['Expert_Count', 'Avg_Dice'])
            
            total_rounds = len(self.phi_library)
            
            for k in range(1, total_rounds + 1):
                current_library = self.phi_library[:k] 
                current_round_id = current_library[-1][0]
                
                metrics = {"dice": []}
                
                for fname in tqdm(test_files, desc=f"Eval Round {current_round_id}"):
                    img_path = os.path.join(TEST_IMG_DIR, fname)
                    gt_path = os.path.join(TEST_GT_DIR, os.path.splitext(fname)[0] + ".png")
                    
                    pil = Image.open(img_path).convert("RGB")
                    img_np = np.array(pil)
                    
                    gt_mask = cv2.imread(gt_path, 0)
                    if gt_mask is not None:
                        gt_mask = cv2.resize(gt_mask, pil.size, interpolation=cv2.INTER_NEAREST)
                        _, gt_bin = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
                    else:
                        gt_bin = None

                    final_mask = None
                    final_r_id = -1
                    best_fallback = None # (mask, r_id, score)
                    min_score = 100.0
                    
                    selection_type = "Fallback"
                    
                    for r_id, text_out in current_library:
                        if r_id in inference_cache[fname]:
                            m_text, unc_score, is_easy = inference_cache[fname][r_id]
                        else:
                            m_text, out = self._predict_mask_raw(pil, text_out)
                            
                            if m_text.sum() == 0:
                                unc_score = 2.0
                                is_easy = False
                            else:  
                                if out["boxes"].shape[0] > 0:
                                    m_box = self._predict_mask_with_pred_boxes(pil, out["boxes"])
                                    iou_c = compute_iou_fg(m_text, m_box)
                                else: iou_c = 0.0
                                
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
                                
                                unc_score =(1 - iou_c) +  ent
                                
                                is_easy = False
                                if unc_score < SOLVED_THRESH and  iou_c > 0.90:
                                    is_easy = True
                            
                            inference_cache[fname][r_id] = (m_text, unc_score, is_easy)
                        
                        if unc_score < min_score:
                            min_score = unc_score
                            best_fallback = (m_text, r_id, unc_score)
                        
                        if is_easy:
                            final_mask = m_text
                            break 
                    
                    if final_mask is None:
                        final_mask, final_r_id, _ = best_fallback
                    
                    d = compute_dice(final_mask, gt_bin)
                    
                    metrics["dice"].append(d)
                    
                    self.visualize_result(fname, pil, final_mask, f"Step_{k}_mask.png")
                    
                avg_d = np.mean(metrics["dice"])
                print(f"Step {k} Result: Dice={avg_d:.4f}")
                writer_curve.writerow([k, avg_d])

        print(f"Visualizations saved to {VIS_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Incremental Sequential Tester")
    parser.add_argument("--test_pool_root", type=str, 
                        default="datasets/Kvasir-seg/query", 
                        help="Directory containing query datasets")

    args = parser.parse_args()

    Test_POOL_ROOT = args.test_pool_root
    TEST_IMG_DIR = os.path.join(Test_POOL_ROOT, "images")
    TEST_GT_DIR = os.path.join(Test_POOL_ROOT, "masks")

    tester = IncrementalSequentialTester()
    tester.run_incremental_inference()