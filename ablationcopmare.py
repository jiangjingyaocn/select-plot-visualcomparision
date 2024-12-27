import os
import json
from PIL import Image, ImageDraw, ImageFont

def load_json(file_path):
    """加载JSON文件并返回数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def get_image_paths(method_config, dataset_config, imglist_config):
    """根据配置文件获取所有图片的路径"""
    image_paths = {}
    # 获取方法的名字，将结果转为列表以便查看
    # 遍历每个数据集
    for dataset in dataset_config['datasetlist']:
        if dataset in imglist_config:
            image_names = imglist_config[dataset]['namelist']
            # method_paths = method_config['GT'][dataset]
            # 构建每张图片在各方法中的路径
            image_paths[dataset] = []
            for name in image_names:
                paths_for_image = []
                for method_name, method_info in method_config.items():
                    path = os.path.join(method_info[dataset]['path'], name + method_info[dataset]['suffix'])
                    paths_for_image.append(path)
                image_paths[dataset].append(paths_for_image)
    return image_paths


from PIL import Image, ImageDraw, ImageFont


def create_comparison_image(image_paths_list, method_names, output_path,
                            spacing=5, method_name_height=20, font_size=30, font_path=None,
                            method_name_spacing=40, line_height_increase=10):
# def create_comparison_image(image_paths_list, method_names, output_path,
#                             spacing=5, method_name_height=0, font_size=56, font_path=None,
#                             method_name_spacing=60, line_height_increase=0):
    """生成视觉对比图并保存"""
    total_height = 0
    opened_images = []
    min_img_width = float('inf')  # 设置一个非常大的初始值

    # 打开所有图片并找到最小宽度
    for dataset in image_paths_list:
        for images_paths in image_paths_list[dataset]:
            row_images = []
            for path in images_paths:
                img = Image.open(path)
                row_images.append(img)
                min_img_width = min(min_img_width, img.width)  # 找到最小宽度
            opened_images.append(row_images)
            total_height += max(img.height for img in row_images)  # 计算总高度

    max_width = min_img_width * len(method_names) + spacing * (len(method_names) - 1)


    font_size = int(max_width / (2304 /56))  # 计算字体大小

    # 设置字体，确保 font_path 是字体文件路径
    if font_path:
        font = ImageFont.truetype(font_path, size=font_size)  # 第一个参数是字体文件路径，第二个参数是字体大小
    else:
        font = ImageFont.load_default()  # 使用默认字体


    y_offset = 0
    # 创建临时图像以计算方法名称的最大高度
    temp_image = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_image)
    max_method_name_height = max(
        [temp_draw.textbbox((0, 0), method_name, font=font)[3] for method_name in method_names])

    # 增加画布高度以适应方法名称的高度
    total_height += spacing * (
                len(image_paths_list) - 1) + max_method_name_height + method_name_spacing + line_height_increase
    # 裁剪画布地步多余的空白（如果没有不用裁），因为用高度最后按照原来的高度计算而不是缩放后的计算。
    total_height = total_height
    # 创建空白图像用于绘制对比图
    # max_width = min_img_width * len(method_names) + spacing * (len(method_names) - 1)
    comparison_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(comparison_image)



    for row_images in opened_images:
        x_offset = 0
        for img in row_images:
            # 按最小宽度缩放图片，高度按比例调整
            scale_factor = min_img_width / img.width
            new_width = min_img_width
            new_height = int(img.height * scale_factor)
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 将调整后的图片粘贴到对比图上
            comparison_image.paste(resized_img, (x_offset, y_offset))
            x_offset += new_width + spacing
        y_offset += new_height + spacing

    y_offset += method_name_spacing

    # 绘制方法名称
    x_offset = 0
    for method_name in method_names:
        print(y_offset)
        bbox = draw.textbbox((0, 0), method_name, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x_offset + (min_img_width - text_width) // 2, y_offset + method_name_height - text_height),
                  method_name, fill=(0, 0, 0), font=font)
        x_offset += min_img_width + spacing
    # 保存对比图
    comparison_image.save(output_path)
    print(f"Comparison image saved at {output_path}")

'''
第一步：读取和解析配置文件 
    从 method.json 中读取每个数据集对应的方法路径和文件后缀。
    从 dataset.json 中读取所需处理的数据集列表。
    从 imglist.json 中读取每个数据集对应的图片名称列表。
'''
# method_file = '../config/article2/compare/ablation.json'
method_file = '/Users/jjy/Desktop/memory/PySODEvalToolkit/config/article1/ablation/ablation_method.json'
# method_file = '../config/article2/compare/copmaremethod_new.json'
# method_file = '../config/article2/compare/comparemethodpipe.json'
# dataset_file = '../config/article2/compare/dataset.json'
dataset_file = '/Users/jjy/Desktop/memory/PySODEvalToolkit/config/article1/ablation/datasetlist.json'
# imglist_file = '../config/article2/compare/imglist.json'
# imglist_file = '../config/article2/compare/selected_images.json'
# imglist_file = '../config/article2/compare/selected_images_new.json'
imglist_file = '/Users/jjy/Desktop/memory/PySODEvalToolkit/config/article1/ablation/selected_images.json'
# 加载配置文件
method_config = load_json(method_file)
dataset_config = load_json(dataset_file)
imglist_config = load_json(imglist_file)
'''
第二步：基于配置文件选择图片路径
根据 dataset.json 中指定的数据集和 imglist.json 中的图片名称，生成method.json中方法对应的图片路径列表。
get_image_paths 函数：
	遍历 dataset.json 中指定的数据集。
	对于每个数据集，从 imglist.json 中获取需要处理的图片名称列表。
	利用 method.json 中的方法路径和文件后缀，构建每张图片在所有方法中的路径列表。
返回值：
	返回一个字典，包含每个数据集的图片路径列表。每张图片对应多个方法的路径。
路径构建：
	使用 os.path.join 来构建文件路径，确保跨平台兼容性。
'''
# 获取图片路径
images_paths = get_image_paths(method_config, dataset_config, imglist_config)
'''
第三步：读取图片并生成视觉对比图
生成视觉对比图：
	创建一个空白画布。
	对每个图片，读取所有方法生成的图片结果，将这些图片横向排列。
	控制每张图片的宽度，使得所有图片在视觉效果上宽度一致。
	将不同图片对应的方法结果排列在不同的行，每一行代表一张图片的所有方法结果。
	在最后一行追加一行文字，显示每一列对应的方法的名字。
'''
# output_path = '../result/article2/visualcompare/comparison_image_output_pipe.png'
output_path = '/Users/jjy/Desktop/memory/PySODEvalToolkit/result/article1/visual/comparison_ablation_1227.pdf'
# output_path = '../result/article2/visualcompare/comparison_image_output.png'
# output_path_pre = '../result/article2/visualcompare/comparison_image_output'



method_names = list(method_config.keys())
font_path = "/System/Library/Fonts/Helvetica.ttc"
create_comparison_image(images_paths, method_names, output_path,font_path=font_path)