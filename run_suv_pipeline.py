import yaml
import subprocess
import os


def run_command(script_path, args):
    cmd = ["python", script_path] + args
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Script {script_path} return {e.returncode}")
        raise

def readtext(file_path):
    with open(file_path, 'r') as f:
        content = f.readline().strip()
    return content

def get_image_class(color_map_path, mask_list):
    """
    从颜色映射文件中获取图像类别
    :param color_map_path: 颜色映射文件路径
    :param mask_list: 目标掩码列表
    """
    with open(color_map_path, 'r') as f:
        color_map = yaml.safe_load(f)

    if not mask_list or len(mask_list) != 1:
        raise ValueError("Expected one target mask, but got multiple or none.")

    target_mask = str(mask_list[0])
    if target_mask not in color_map:
        raise ValueError(f"Mask {target_mask} not found in color map.")

    return color_map[target_mask]

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

    # pretrained models
    og_export = cfg['og_export']
    sp_export = cfg['sp_export']
    dino_export = cfg['dino_export']
    lg_export = cfg['lg_export']
    color_map_path = os.path.join(base, cfg['color_map_path'])

    image_folder = os.path.join(base, folder)
    mask_folder = os.path.join(base, f"{folder}_masks")
    
    # output paths
    class_name = ""
    class_txt_path = ""
    anchor_file = os.path.join(output_folder, "anchor.txt") # to put anchor frame index
    kpt_folder = os.path.join(output_folder, "kpts")     # to put keypoints

    # region === 0. Specify the type of instruction ===
    instruction_cfg = cfg['instruction_config']
    turn_on_instruction = instruction_cfg.get('turn_on', True)
    text_path = instruction_cfg['text_path']
    ref_image_path = instruction_cfg['image_path']
    ref_mask_path = instruction_cfg['mask_path']
    if turn_on_instruction:
        instruction_type = instruction_cfg['type'].lower()

        if instruction_type == 'text':
            # === Get target class for text input ===
            model_cfg = cfg['language_model_config']
            text_model_type = model_cfg['Model_type'].lower()
            model_name = model_cfg['Model_name']
            instruction_file = text_path
            turn_on_lang = model_cfg.get('turn_on', True)
            if turn_on_lang: 
                if text_model_type == "manual_class":
                    class_name = model_cfg['Class_name']
                    manual_path = os.path.join(output_folder, "seg_target_cat.txt")
                    with open(manual_path, 'w') as f:
                        f.write(class_name)
                    print(f"[Manual Class] was specified as '{class_name}'，write to {manual_path}")
                    class_txt_path = manual_path
                elif text_model_type in ["llm", "medical_llm", "minillm"]:
                    script_map = {
                        "llm": "src/1.match_desc_llm.py",
                        "medical_llm": "src/1.match_desc_llm_med.py",
                        "minillm": "src/1.match_desc_vit.py"
                    }
                    script = script_map[text_model_type]
                    desc_dir = os.path.join(support, "expanded_descs")
                    class_txt_path = os.path.join(output_folder, f"seg_target_cat.txt")
                    api_key = ""
                    if model_name == "gemini":
                        api_key = model_cfg['gemini_api_key']
                    if model_name == "gpt":
                        api_key = model_cfg['gpt_api_key']
                    run_command(script, [
                        "--desc_dir", desc_dir,
                        "--instruction", instruction_file,
                        "--model_name", model_name,
                        "--api_key", api_key,
                        "--output_path", class_txt_path
                    ])
                    class_name = readtext(class_txt_path).strip()
                else:
                    raise ValueError(f"Unsupported model_type: {text_model_type}")
            assert(os.path.exists(class_txt_path)), \
                    f"The language model was not turned on. Check the config." 

    assert(os.path.exists(class_txt_path) or (os.path.exists(ref_image_path) \
    and os.path.exists(ref_mask_path))), \
    f"Class text file {class_txt_path} does not exist. Please run instruction step first."
    print(class_txt_path)
    # endregion
    
    # region !! 还没写step的 === 0. Anchor Frame Selection === 
    print(" select anchor frames")
    anchor_cfg = cfg['anchor_frame_config']
    anchor_type = anchor_cfg['anchor_type'].lower()
    turn_on_anchor = anchor_cfg.get('turn_on', True)
    if turn_on_anchor:
        if anchor_type not in ['ssim1', 'ssim2', 'step1', 'step5', 'step10']:
            raise ValueError(f"Unsupported anchor_type: {anchor_type}")
        if anchor_type == 'ssim1':
            run_command("src/0.anchor_frame_info_first.py", [
                "--folder", image_folder,
                "--top_percent", str(anchor_cfg['top_percent']),
                "--ssim_diff_threshold", str(anchor_cfg['ssim_diff_threshold']),
                "--max_keyframes", str(anchor_cfg['max_keyframes']),
                "--use_percentile", str(anchor_cfg['use_percentile']),
                "--output_path", anchor_file,
            ])
        elif anchor_type == 'ssim2':
            run_command("src/0.anchor_frame_diff_first.py", [
                "--folder", image_folder,
                "--top_percent", str(anchor_cfg['top_percent']),
                "--ssim_diff_threshold", str(anchor_cfg['ssim_diff_threshold']),
                "--max_keyframes", str(anchor_cfg['max_keyframes']),
                "--use_percentile", str(anchor_cfg['use_percentile']),
                "--output_path", anchor_file,
            ])
        else:
            pass
    else:
        assert(os.path.exists(anchor_file)), \
               f"Anchor file {anchor_file} does not exist. Please run anchor frame selection first."
        
    ref_img = ""
    ref_mask = ""
    # endregion

    # region === 3. locate optimal support image ===
    locate_cfg = cfg['locate_config']
    turn_on_locate = locate_cfg.get('turn_on', True)
    if not turn_on_locate:
        assert(instruction_type == 'image'), \
            f"Please run locate support image step first."
        if instruction_type == 'image':
            ref_img = instruction_cfg['image_path']
            ref_mask = instruction_cfg['mask_path']
    else:
        if instruction_type == 'image':
            ref_img = instruction_cfg['image_path']
            ref_mask = instruction_cfg['mask_path']
        else:
            locate_method = locate_cfg['locate_method'].lower()
            if locate_method == 'label':
                ref_img, ref_mask = None
            elif  locate_method == 'label+img':
                ref_img, ref_mask = None
    # endregion

    # region === 2. Initial Anchor Descriptors ===
    print("generate anchor descriptors")
    anchor_desc_cfg = cfg['anchor_kpts_config']
    descriptor_type = anchor_desc_cfg['descriptor_type'].lower()
    turn_on_descriptor = anchor_desc_cfg.get('turn_on', True)
    if turn_on_descriptor:
        
        if descriptor_type not in ['random', 'omniglue', 'superpoint']:
            raise ValueError(f"Unsupported descriptor_type: {descriptor_type}")
        if descriptor_type == 'random':
            assert(os.path.exists(og_export) and os.path.exists(sp_export) \
                     and os.path.exists(dino_export)), \
                 f"Please check the paths of og_export, sp_export, dino_export in config."
            run_command("src/2.generate_anchor_rd_kpts.py", [
                "--image_folder", image_folder,
                "--anchor_file", anchor_file,
                "--output", kpt_folder,
                "--og_export", og_export,
                "--sp_export", sp_export,
                "--dino_export", dino_export,
                "--ssim_min", str(anchor_desc_cfg.get('ssim_min', 0.8)),
                "--ssim_max", str(anchor_desc_cfg.get('ssim_max', 0.95))
            ])

        elif descriptor_type == 'omniglue':
            assert(os.path.exists(og_export) and os.path.exists(sp_export) \
                     and os.path.exists(dino_export)), \
                 f"Please check the paths of og_export, sp_export, dino_export in config."
            run_command("src/2.generate_anchor_kpts.py", [
                "--image_folder", image_folder,
                "--anchor_file", anchor_file,
                "--output", kpt_folder,
                "--og_export", og_export,
                "--sp_export", sp_export,
                "--dino_export", dino_export,
                "--ssim_min", str(anchor_desc_cfg.get('ssim_min', 0.8)),
                "--ssim_max", str(anchor_desc_cfg.get('ssim_max', 0.95))
            ])
        
        if descriptor_type == 'superpoint':
            pass
    assert(os.path.exists(kpt_folder)), f"Keypoint folder {kpt_folder} does not exist. \
                Please run anchor descriptor generation first." 
    import ast
    # === 3. Classify Points ===
    print("classify kpts")
    if class_name == "":
        if instruction_type == 'text':
            class_name = readtext(class_txt_path).strip() if class_txt_path else ""
            print(f"[Classify Points] 从 {class_txt_path} 读取到目标类别：'{class_name}'")
        elif instruction_type == 'image':
            mask_list = instruction_cfg['target_mask']
            print(mask_list, type(mask_list))
            wrapped = f"[{mask_list}]"
            mask_color_list = ast.literal_eval(wrapped)
            print(len(mask_color_list))
            if len(mask_color_list) != 1:
                raise ValueError(f"Expected one target mask, but got {len(mask_color_list)} masks.")
            class_name = get_image_class(color_map_path, mask_color_list)

    classify_cfg = cfg['classify_kps']
    turn_on_kpts = classify_cfg.get('turn_on', True)

    method = classify_cfg['class_method'].lower()
    click_folder = os.path.join(output_folder, class_name, cfg['click_folder'])

    script_map = {
        "omniglue": "src/3.classify_kpts_omniglue.py",
        "lightglue": "src/3.classify_kpts_lightglue.py",
        "cluster": "src/3.classify_points_cluster.py",
        "point": "src/3.classify_points_sig_pt.py",
        "cross_attn": "src/3.classify_kpts_cross_attn.py",
        "histogram": "src/3.classify_kpts_histogram.py",
        "matching": "src/3.classify_kpts_matching.py",

    }


    if not turn_on_kpts:
        assert(os.path.exists(kpt_folder)), \
            f"Keypoint folder {kpt_folder} does not exist. Please run classify points step first."
        return
    else:
        if method not in script_map:
            raise ValueError(f"Unsupported class_method: {method}")
        subregion = classify_cfg['subregion']
        common_args = [
            "--image_folder", image_folder,
            "--ref_img_path", ref_img,
            "--ref_mask_path", ref_mask,
            "--color_map_path", color_map_path,
            "--click_save_folder", click_folder,
            "--anchor_file", anchor_file,
            "--target_class", class_name,
            "--kpt_folder", kpt_folder,
            "--subregion", str(subregion),
        ]

        if method == "omniglue":
            extra_args = [
                "--og_export", og_export,
                "--sp_export", sp_export,
                "--dino_export", dino_export,
            ]
            run_command(script_map[method], common_args + extra_args)

        elif method == "lightglue":
            extra_args = [
                "--lg_export", lg_export,
                "--sp_export", sp_export,
            ]
            run_command(script_map[method], common_args + extra_args)

        elif method == "cross_attn":
            feature_json_folder = os.path.join(base,classify_cfg['feature_json_folder'])
            run_command(script_map[method], [
                "--image_folder", image_folder,
                "--match_txt_folder", kpt_folder,
                "--feature_json_folder", feature_json_folder,
                "--click_save_folder", kpt_folder,
                "--target_class", class_name
            ])
        elif method == "histogram":
            support_img_path = os.path.join(base, classify_cfg['support_img_path'])
            support_mask_path = os.path.join(base, classify_cfg['support_mask_path'])
            #label_map_json = os.path.join(base, classify_cfg['label_map_json'])  # e.g., suv.json
            run_command(script_map[method], [
                "--image_folder", image_folder,
                "--match_txt_folder", kpt_folder,
                "--support_img_path", support_img_path,
                "--support_mask_path", support_mask_path,
                "--click_save_folder", kpt_folder,
                "--target_class", class_name,
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
                "--click_save_folder", kpt_folder,
                "--target_class", class_name,
                "--color_map_path", color_map_path,
                "--index_file", anchor_file,
                "--match_kpt_folder", kpt_folder
            ])

        else:
            # 其他通用分类方法
            feature_path = os.path.join(base,classify_cfg['feature_json_path'])
            run_command(script_map[method], [
                "--image_folder", image_folder,
                "--match_txt_folder", kpt_folder,
                "--feature_json_path", feature_path,
                "--click_save_folder", click_folder,
                "--target_class", class_name
            ])

    # 计算有多少个point会落入gt_正确的mask
    if classify_cfg.get('evaluate', True):
        save_json_path = os.path.join(cfg['output_root'], folder, class_name, "point_hit_stats.json")

        run_command("src/1.5.check_points_in_gt_mask.py", [
            "--base_dataset_path", cfg['base_dataset_path'],
            "--folder_name", folder,
            "--click_folder", click_folder,
            "--save_path", save_json_path,
            "--class_name", cfg["language_model_config"]["Class_name"],
            "--suv_json_path", cfg["color_map_path"],

        ])        
    else:
        print("skip evaluate points in gt mask")

    result_folder = os.path.join(output_folder, class_name, "result")
    
    # === 4. SUV Auto Segmentation ===
    suv_cfg = cfg['suv_auto']
    turn_on_suv = suv_cfg.get('turn_on', True)
    if turn_on_suv:
        run_command("suv_auto.py", [
            "--video_dir", image_folder,
            "--video_result", result_folder,
            "--click_folder", click_folder,
            "--seg_model", suv_cfg['seg_model'],
            "--checkpoint", suv_cfg['sam2_checkpoint'],
            "--model_cfg", suv_cfg['model_cfg'],
            "--prompt_type", suv_cfg['prompt_type'],
            "--augmentation", suv_cfg['augmentation'],
            "--target_class", class_name,
        ])
    
    

if __name__ == "__main__":
    main("suv.yaml")
