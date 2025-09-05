#!/usr/bin/env python3
import os, argparse, json
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import cv2
import pytesseract
import pandas as pd

def preprocess_pil(pil_img, scale=1.6, contrast=1.6, sharpness=1.2):
    img = ImageOps.grayscale(pil_img)
    w,h = img.size
    img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img

def image_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def detect_table_cells(cv_img_gray):
    thresh = cv2.adaptiveThreshold(cv_img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,9)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,30))
    hor = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
    ver = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=2)
    grid = cv2.add(hor, ver)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w < 50 or h < 30:
            continue
        boxes.append((x,y,w,h))
    if not boxes:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if w < 40 or h < 12:
                continue
            boxes.append((x,y,w,h))
    boxes_sorted = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    table_box = boxes_sorted[0] if boxes_sorted else None
    cells = []
    if table_box:
        tx,ty,tw,th = table_box
        table_roi = thresh[ty:ty+th, tx:tx+tw]
        horiz_proj = np.sum(table_roi, axis=1)
        vert_proj = np.sum(table_roi, axis=0)
        hp = (horiz_proj - horiz_proj.min()) / (horiz_proj.max() - horiz_proj.min() + 1e-9)
        vp = (vert_proj - vert_proj.min()) / (vert_proj.max() - vert_proj.min() + 1e-9)
        row_gaps = np.where(hp < 0.05)[0]
        col_gaps = np.where(vp < 0.05)[0]
        def cluster_indices(idxs, tol=6):
            groups = []
            if len(idxs)==0: return groups
            cur = [idxs[0]]
            for i in idxs[1:]:
                if i - cur[-1] <= tol:
                    cur.append(i)
                else:
                    groups.append(cur)
                    cur = [i]
            groups.append(cur)
            return [int(round(sum(g)/len(g))) for g in groups]
        row_seps = cluster_indices(row_gaps, tol=8)
        col_seps = cluster_indices(col_gaps, tol=8)
        row_coords = [0] + row_seps + [table_roi.shape[0]-1]
        col_coords = [0] + col_seps + [table_roi.shape[1]-1]
        row_coords = sorted(list(dict.fromkeys(row_coords)))
        col_coords = sorted(list(dict.fromkeys(col_coords)))
        for r in range(len(row_coords)-1):
            r0 = row_coords[r]; r1 = row_coords[r+1]
            if r1 - r0 < 10: continue
            for c in range(len(col_coords)-1):
                c0 = col_coords[c]; c1 = col_coords[c+1]
                if c1 - c0 < 20: continue
                x = tx + max(c0-2,0); y = ty + max(r0-2,0)
                w = (c1 - c0) + 4; h = (r1 - r0) + 4
                cells.append((x,y,w,h))
    else:
        cells = boxes_sorted
    cells_sorted = sorted(cells, key=lambda b:(b[1], b[0]))
    return cells_sorted

def ocr_image_region(pil_img):
    pre = preprocess_pil(pil_img, scale=1.0, contrast=1.4, sharpness=1.0)
    txt = pytesseract.image_to_string(pre, config="--oem 3 --psm 6")
    return txt.strip()

def run_ocr_on_image(image_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pil = Image.open(image_path).convert("RGB")
    pre = preprocess_pil(pil)
    cv = image_to_cv(pre)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    cells = detect_table_cells(gray)
    debug_img = cv.copy()
    rows = []
    if cells:
        for (x,y,w,h) in cells:
            cv2.rectangle(debug_img, (int(x),int(y)), (int(x+w),int(y+h)), (0,255,0), 1)
        rows_cells = []
        cells_sorted = sorted(cells, key=lambda b: (b[1], b[0]))
        current_row = [cells_sorted[0]]
        for c in cells_sorted[1:]:
            if abs(c[1] - current_row[-1][1]) <= max(10, int(0.2*current_row[-1][3])):
                current_row.append(c)
            else:
                rows_cells.append(current_row)
                current_row = [c]
        if current_row:
            rows_cells.append(current_row)
        table_text = []
        for r in rows_cells:
            r_sorted = sorted(r, key=lambda b: b[0])
            row_text = []
            for (x,y,w,h) in r_sorted:
                crop = pre.crop((x,y,x+w,y+h))
                txt = ocr_image_region(crop)
                row_text.append(txt)
            table_text.append(row_text)
    else:
        fulltxt = pytesseract.image_to_string(pre, config="--oem 3 --psm 6")
        table_text = [line for line in [l.strip() for l in fulltxt.splitlines()] if line]
        table_text = [[r] for r in table_text]
    base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(out_dir, base + ".json")
    csv_path = os.path.join(out_dir, base + ".csv")
    html_path = os.path.join(out_dir, base + ".html")
    debug_path = os.path.join(out_dir, base + "_debug.png")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"source": image_path, "table": table_text}, f, ensure_ascii=False, indent=2)
    try:
        df = pd.DataFrame(table_text)
        df.to_csv(csv_path, index=False, header=False)
        df.to_html(html_path, index=False, header=False)
    except Exception as e:
        with open(csv_path, "w", encoding="utf-8") as f:
            for row in table_text:
                f.write(",".join(row) + "\n")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><body><pre>" + "\\n".join([','.join(r) for r in table_text]) + "</pre></body></html>")
    cv2.imwrite(debug_path, debug_img)
    print("Outputs written to:", out_dir)
    print("JSON:", json_path)
    print("CSV:", csv_path)
    print("HTML:", html_path)
    print("Debug image (cells):", debug_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image or PDF input path")
    parser.add_argument("--output", required=True, help="Output directory for JSON/CSV/HTML")
    args = parser.parse_args()
    run_ocr_on_image(args.input, args.output)
