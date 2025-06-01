import yaml
import subprocess
import os

def run_command(script_path, args):
    cmd = ["python", script_path] + args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 基础路径配置
    base = cfg['base_dataset_path']
    folder = cfg['folder_name']
    support = os.path.join(base, cfg.get('support_subdir', "support"))
    output_root = cfg.get('output_root', './Outputs')
    output_folder = os.path.join(output_root, folder)
    os.makedirs(output_folder, exist_ok=True)

    # 构建通用路径
    image_folder = os.path.join(base, folder)
    mask_folder = os.path.join(base, f"{folder}_masks")
    anchor_file = os.path.join(output_folder, "anchor.txt")
    matched_kpt_folder = os.path.join(output_folder, "matched_kpts")
        
    # === 0. Anchor Frame Selection ===
    anchor_cfg = cfg['anchor_frame_config']
    run_command("src/0.anchor_frame.py", [
        "--folder", image_folder,
        "--top_percent", str(anchor_cfg['top_percent']),
        "--ssim_diff_threshold", str(anchor_cfg['ssim_diff_threshold']),
        "--max_keyframes", str(anchor_cfg['max_keyframes']),
        "--use_percentile", str(anchor_cfg['use_percentile'])
    ])

    # === 1. Match Description ===
    model_cfg = cfg['language_model_config']
    model_type = model_cfg['Model_type'].lower()
    model_name = model_cfg['Model_name']
    class_name = model_cfg['Class_name']
    instruction_file = model_cfg['Instruction']
    class_txt_path = ""
    if model_type == "manual_class":
        manual_path = os.path.join(output_folder, "seg_target_category_manual.txt")
        class_txt_path = manual_path
        with open(manual_path, 'w') as f:
            f.write(class_name)
        print(f"[Manual Class] 指定类别为 '{class_name}'，已写入 {manual_path}")

    elif model_type in ["llm", "medical_llm", "sbert"]:
        script_map = {
            "llm": "src/1.match_desc_llm.py",
            "medical_llm": "src/1.match_desc_llm_med.py",
            "sbert": "src/1.match_desc_vit.py"
        }
        script = script_map[model_type]
        desc_dir = os.path.join(support, "expanded_descs")

        run_command(script, [
            "--folder", folder,
            "--desc_dir", desc_dir,
            "--instruction", instruction_file,
            "--model_name", model_name,
        ])
        class_txt_path = os.path.join(output_folder, folder, f"seg_target_category_{model_name}.txt")

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    

       
    # === 2. Initial Anchor Descriptors ===
    anchor_desc_cfg = cfg['anchor_descriptors_config']
    run_command("src/2.generate_anchor_kpts.py", [
        "--image_folder", image_folder,
        "--index_file", anchor_file,
        "--output", matched_kpt_folder,
        "--og_export", anchor_desc_cfg['og_export'],
        "--sp_export", anchor_desc_cfg['sp_export'],
        "--dino_export", anchor_desc_cfg['dino_export'],
        "--ssim_min", str(anchor_desc_cfg.get('ssim_min', 0.8)),
        "--ssim_max", str(anchor_desc_cfg.get('ssim_max', 0.95))
    ])
    

    class_txt_path = os.path.join(output_folder, "seg_target_category_manual.txt")
        
    # === 3. Classify Points ===
    with open(class_txt_path, 'r') as f:
        matched_class_name = f.readline().strip()
        print(f"[Classify Points] 从 {class_txt_path} 读取到目标类别：'{matched_class_name}'")
    click_folder = os.path.join(output_folder, matched_class_name,"clicks")

    classify_cfg = cfg['classify_kps']
    method = classify_cfg['class_method'].lower()
    script_map = {
        "cluster": "src/3.classify_points_cluster.py",
        "point": "src/3.classify_points_sig_pt.py",
        "cross_attn": "src/3.classify_kpts_cross_attn.py",
        "histogram": "src/3.classify_kpts_histogram.py",
        "matching": "src/3.classify_kpts_matching.py",
        "omniglue": "src/3.classify_kpts_omniglue.py",
        "lightglue": "src/3.classify_kpts_lightglue.py",
    }

    if method not in script_map:
        raise ValueError(f"Unsupported class_method: {method}")

    color_map_path = os.path.join(base, classify_cfg['color_map_path'])

    # 特例处理 omniglue 模式
    if method == "omniglue":
        support_img_path = os.path.join(support, classify_cfg['support_img_path'])
        support_mask_path = os.path.join(support, classify_cfg['support_mask_path'])
        run_command(script_map[method], [
            "--image_folder", image_folder,
            "--support_img_path", support_img_path,
            "--support_mask_path", support_mask_path,
            "--color_map_path", color_map_path,
            "--click_save_folder", click_folder,
            "--index_file", anchor_file,
            "--target_class", matched_class_name
        ])
    elif method == "cross_attn":
        feature_json_folder = os.path.join(base,classify_cfg['feature_json_folder'])
        run_command(script_map[method], [
            "--image_folder", image_folder,
            "--match_txt_folder", matched_kpt_folder,
            "--feature_json_folder", feature_json_folder,
            "--click_save_folder", click_folder,
            "--target_class", matched_class_name
        ])
    elif method == "histogram":
        support_img_path = os.path.join(base, classify_cfg['support_img_path'])
        support_mask_path = os.path.join(base, classify_cfg['support_mask_path'])
        #label_map_json = os.path.join(base, classify_cfg['label_map_json'])  # e.g., suv.json
        run_command(script_map[method], [
            "--image_folder", image_folder,
            "--match_txt_folder", matched_kpt_folder,
            "--support_img_path", support_img_path,
            "--support_mask_path", support_mask_path,
            "--click_save_folder", click_folder,
            "--target_class", matched_class_name,
            "--label_map_json", color_map_path
        ])   
    elif method == "matching":
        support_label_dir = os.path.join(support, 'labels')
        support_img_dir = os.path.join(support, 'imgs')
        support_mask_dir = os.path.join(support, 'mask')
        run_command(script_map[method], [
            "--image_folder", image_folder,
            "--support_label_dir", support_label_dir,
            "--support_img_dir", support_img_dir,
            "--support_mask_dir", support_mask_dir,
            "--click_save_folder", click_folder,
            "--target_class", matched_class_name,
            "--color_map_path", color_map_path,
            "--index_file", anchor_file,
            "--match_kpt_folder", matched_kpt_folder
        ])
    elif method == "lightglue":
        support_img_path = os.path.join(support, classify_cfg['support_img_path'])
        support_mask_path = os.path.join(support, classify_cfg['support_mask_path'])
        run_command(script_map[method], [
            "--image_folder", image_folder,
            "--support_img_path", support_img_path,
            "--support_mask_path", support_mask_path,
            "--color_map_path", color_map_path,
            "--click_save_folder", click_folder,
            "--index_file", anchor_file,
            "--target_class", matched_class_name
        ])
    else:
        # 其他通用分类方法
        feature_path = os.path.join(base,classify_cfg['feature_json_path'])
        run_command(script_map[method], [
            "--image_folder", image_folder,
            "--match_txt_folder", matched_kpt_folder,
            "--feature_json_path", feature_path,
            "--click_save_folder", click_folder,
            "--target_class", matched_class_name
        ])

    # 计算有多少个point会落入gt_正确的mask

    save_json_path = os.path.join(cfg['output_root'], folder, class_name, "point_hit_stats.json")

    run_command("src/1.5.check_points_in_gt_mask.py", [
        "--base_dataset_path", cfg['base_dataset_path'],
        "--folder_name", folder,
        "--click_folder", click_folder,
        "--save_path", save_json_path,
        "--class_name", cfg["language_model_config"]["Class_name"],
        "--suv_json_path", cfg["color_map_path"],

    ])        

    result_folder = os.path.join(output_folder, class_name, "result")
    """
    # === 4. SUV Auto Segmentation ===
    suv_cfg = cfg['suv_auto']
    run_command("suv_auto.py", [
        "--video_dir", image_folder,
        "--video_result", result_folder,
        "--click_folder", click_folder,
        "--seg_model", suv_cfg['seg_model'],
        "--checkpoint", suv_cfg['sam2_checkpoint'],
        "--model_cfg", suv_cfg['model_cfg'],
        "--prompt_type", suv_cfg['prompt_type'],
        "--augmentation", suv_cfg['augmentation'],
        "--target_class", matched_class_name,
    ])
    """
    

if __name__ == "__main__":
    main("suv.yaml")
