import os
import json
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from py_sod_metrics import Smeasure, MAE, WeightedFmeasure
from skimage.color import rgb2gray
def check_image_channels(path, dataset_name):
    img = mpimg.imread(path)
    # 获取图片的通道数
    if img.ndim == 2:
        channels = 1  # 灰度图
    elif img.ndim == 3:
        channels = img.shape[2]  # 彩色图，获取通道数
    else:
        channels = 0  # 非标准格式
        # 检查通道数是否符合要求
    if channels not in [1, 3]:
        print(f"警告: 图像 '{os.path.basename(path)}' (数据集: '{dataset_name}') 通道数为 {channels}, 需要手动检查!")
        print(f"路径: {path}\n")

def prepare_data(pred, gt):
    # 如果pred是彩色图像，检查是否为RGBA格式
    if pred.ndim == 3 and pred.shape[2] == 4:
        pred = pred[..., :3]  # 去掉透明度通道，保留RGB通道
    if pred.ndim == 3:
        pred = rgb2gray(pred)

    # 如果gt是彩色图像，检查是否为RGBA格式
    if gt.ndim == 3 and gt.shape[2] == 4:
        gt = gt[..., :3]  # 去掉透明度通道，保留RGB通道
    if gt.ndim == 3:
        gt = rgb2gray(gt)
    # 如果 gt 和 pred 的形状不同，调整 pred 的形状
    if pred.shape != gt.shape:
        # 将 pred 调整为与 gt 相同的大小
        pred = resize(pred, gt.shape, mode='reflect', anti_aliasing=True)
    return pred, gt

def load_json(file_path):
    """加载JSON文件并返回数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_image_paths(method_config, dataset_config, imglist_config):
    """根据配置文件获取所有图片的路径"""
    image_paths = {}
    # 遍历每个数据集
    for dataset in dataset_config['datasetlist']:
        if dataset in imglist_config:
            image_names = imglist_config[dataset]['namelist']
            image_paths[dataset] = []
            for name in image_names:
                paths_for_image = []
                for method_name, method_info in method_config.items():
                    path = os.path.join(method_info[dataset]['path'], name + method_info[dataset]['suffix'])
                    paths_for_image.append(path)
                image_paths[dataset].append(paths_for_image)
    return image_paths

def display_image_row(image_paths, dataset_name):
    """
    显示一行图片，并在画布上显示数据集名称和其中一张图片的路径
    """
    # 设置画布大小
    fig, ax = plt.subplots(1, len(image_paths), figsize=(15, 5))
    # 遍历每张图片路径并显示图片
    for i, path in enumerate(image_paths):
        img = mpimg.imread(path)
        ax[i].imshow(img)
        ax[i].axis('off')
        ax[i].set_title(f'Image {i + 1}')
    # 在画布上显示数据集名称和第一张图片的路径
    plt.suptitle(f"Dataset: {dataset_name}\nExample Image Path: {image_paths[0]}")
    # 显示图像
    plt.show()
    # 关闭图像窗口
    plt.close(fig)


def meets_criteria_ours(pred: np.ndarray, gt: np.ndarray) -> bool:
    # 计算 Smeasure
    smeasure_value = calculate_smeasure(pred, gt)
    wfm = calculate_wfmeasure(pred,gt)
    # 计算 MAE
    mae_value = calculate_mae(pred, gt)

    # 检查条件是否满足
    # if smeasure_value > 0.95:
    # if smeasure_value > 0.95 and mae_value < 0.15:
    # if smeasure_value > 0.85 and wfm > 0.7:
    if wfm > 0.9 :
    # if wfm < 0.3 and mae_value > 0.6:
        return True
    else:
        return False

def meets_criteria_others(pred: np.ndarray, gt: np.ndarray) -> bool:
    # 计算 Smeasure
    # smeasure_value = calculate_smeasure(pred, gt)
    wfm = calculate_wfmeasure(pred,gt)
    # 计算 MAE
    # mae_value = calculate_mae(pred, gt)
    # 检查条件是否满足
    # if smeasure_value < 0.9 and mae_value > 0.2  and smeasure_value > 0.5 and  mae_value < 0.6:
    # if smeasure_value < 0.9 and wfm > 0.6 :
    if wfm < 0.85:
    # if wfm < 0.2:
        return True
    else:
        return False

# 计算Smeasure值的方法
def calculate_smeasure(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
    smeasure = Smeasure(alpha=alpha)
    smeasure.step(pred=pred, gt=gt)
    results = smeasure.get_results()
    sm_value = results['sm']
    return sm_value

def calculate_wfmeasure(pred: np.ndarray, gt: np.ndarray, beta: float = 1) -> float:
    wfmeasure = WeightedFmeasure(beta=beta)
    wfmeasure.step(pred=pred, gt=gt)
    results = wfmeasure.get_results()
    wfm_value = results['wfm']
    return wfm_value

def calculate_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = resize(pred, gt.shape, anti_aliasing=True)
    mae_metric = MAE()
    mae_metric.step(pred=pred, gt=gt)
    results = mae_metric.get_results()
    mae_value = results['mae']
    return mae_value

def process_images(image_paths, dataset_name,selected_images):
    """
    对图像列表中的图像进行Smeasure和MAE计算，并将符合条件的图像文件名保存到文件中
    """
    # 真值在索引i=1, 我们的方法在索引i=2
    gt_image = mpimg.imread(image_paths[1])
    our_method_image = mpimg.imread(image_paths[2])

    our_method_image, gt_image = prepare_data(our_method_image, gt_image)
    if meets_criteria_ours(our_method_image, gt_image):
        # 设置计数器，满足条件的其他方法到达一定数量，则将这个图片加入到选择列表中。
        criteria_met_count = 0
        required_criteria_met = 3        # 设置至少5张图片满足条件

        for i, other_image_path in enumerate(image_paths):
            if i==0 or i == 1 or i == 2:  # 跳过真值和我们的方法和原图片
                 continue
            other_image = mpimg.imread(other_image_path)
            other_image, gt_image = prepare_data(other_image, gt_image)
            # 如果当前方法满足条件，计数加1
            if meets_criteria_others(other_image, gt_image):
                criteria_met_count += 1
            # 如果满足条件的图片数达到要求，跳出循环
            if criteria_met_count >= required_criteria_met:
                break
            # 如果满足条件的图片数达到要求，则记录图片信息
        if criteria_met_count >= required_criteria_met:
            filename_with_extension = os.path.basename(image_paths[2])  # 获取文件名（带扩展名）
            filename_without_extension = os.path.splitext(filename_with_extension)[0]
            selected_images.append((dataset_name, filename_without_extension))
            print(dataset_name, filename_without_extension, len(selected_images))
            print(selected_images)
# 加载配置文件
# method_file = '../config/article2/compare/copmaremethod.json'
# method_file='/Volumes/data/MASTER degree candidate/Article/article1/PySODEvalToolkit/config/compare/method.json'
method_file='/Volumes/data/MASTER degree candidate/Article/article1/PySODEvalToolkit/config/article1/ablation/ablation_method.json'
# method_file = '../config/article2/compare/copmaremethod_new.json'
dataset_file = '/Volumes/data/MASTER degree candidate/Article/article1/PySODEvalToolkit/config/article1/compare/dataset.json'
imglist_file = '/Volumes/data/MASTER degree candidate/Article/article1/PySODEvalToolkit/config/article1/compare/imglist.json'
# imglist_file = '../config/article2/compare/selected_images.json'
output_file = '/Volumes/data/MASTER degree candidate/Article/article1/PySODEvalToolkit/config/article1/compare/selected_images_ablation_visual.json'
method_config = load_json(method_file)
dataset_config = load_json(dataset_file)
imglist_config = load_json(imglist_file)
# 获取图片路径
images_paths = get_image_paths(method_config, dataset_config, imglist_config)
selected_images = []
# 展示每个数据集中的所有图片
for dataset_name in images_paths.keys():
    for image_set in images_paths[dataset_name]:
        process_images(image_set, dataset_name,selected_images)
        # display_image_row(image_set, dataset_name)
results = {}
# 构建结果字典
for dataset_name, filename in selected_images:
    if dataset_name not in results:
        results[dataset_name] = {"namelist": []}
    results[dataset_name]["namelist"].append(filename)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)
# 输出检查
print(json.dumps(results, indent=4))