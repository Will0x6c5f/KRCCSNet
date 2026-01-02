import numpy as np
import os
import glob
from time import time
import cv2
from skimage.metrics import structural_similarity as ssim
import argparse
import model.krccsnet as krccsnet
import warnings
import torch

warnings.filterwarnings("ignore")


def rgb_to_ycbcr(input_tensor):
    output = torch.zeros_like(input_tensor)

    r = input_tensor[:, 0, :, :]
    g = input_tensor[:, 1, :, :]
    b = input_tensor[:, 2, :, :]

    output[:, 0, :, :] = 65.481 * r + 128.553 * g + 24.966 * b + 16.0
    output[:, 1, :, :] = -37.797 * r - 74.203 * g + 112.0 * b + 128.0
    output[:, 2, :, :] = 112.0 * r - 93.786 * g - 18.214 * b + 128.0

    return output

def ycbcr_to_rgb(input_tensor):

    output = torch.zeros_like(input_tensor)
    
    y = input_tensor[:, 0, :, :]
    cb = input_tensor[:, 1, :, :]
    cr = input_tensor[:, 2, :, :]
    
    y_shifted = y - 16.0
    cb_shifted = cb - 128.0
    cr_shifted = cr - 128.0
    
    output[:, 0, :, :] = 0.00456621 * y_shifted + 0.00625893 * cr_shifted
    output[:, 1, :, :] = 0.00456621 * y_shifted - 0.00153632 * cb_shifted - 0.00318811 * cr_shifted
    output[:, 2, :, :] = 0.00456621 * y_shifted + 0.00791071 * cb_shifted
    
    return output

def psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))



def main():
    global args
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 构建模型
    model = krccsnet.build_krccsnet(sensing_rate=args.sensing_rate)
    model = model.cuda()

    # 加载权重
    dic = './saved_model/' + str(args.model) + '_' + str(args.sensing_rate) + '.pth'
    state_dict = torch.load(dic)
    if state_dict.__class__ == ().__class__: # Handle tuple case if necessary
        state_dict = state_dict[0]
    model.load_state_dict(state_dict)
    model.eval() # 设置为评估模式

    # 准备测试文件
    ext = {'/*.jpg', '/*.png', '/*.tif'}
    filepaths = []
    test_dir = os.path.join('BSDS500/set14', args.test_name)
    for img_type in ext:
        filepaths = filepaths + glob.glob(test_dir + img_type)

    # 结果保存路径
    result_dir = os.path.join(args.result_dir, args.test_name, str(args.sensing_rate))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    Time_All = np.zeros([1, ImgNum], dtype=np.float32)

    print("\nCS Reconstruction Start")

    with torch.no_grad():
        for img_no in range(ImgNum):
            imgName = filepaths[img_no]
            

            Img_bgr = cv2.imread(imgName, 1)
            h, w = Img_bgr.shape[:2]
            new_h = h - (h % 16) 
            new_w = w - (w % 16)
            Img_bgr = Img_bgr[:new_h, :new_w, :]

          
            Img_rgb = cv2.cvtColor(Img_bgr, cv2.COLOR_BGR2RGB)
            
     
            input_tensor = torch.from_numpy(Img_rgb.transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 255.0
            
   
            ycbcr_tensor = rgb_to_ycbcr(input_tensor)
            

            y_channel = ycbcr_tensor[:, 0:1, :, :]
            cb_channel = ycbcr_tensor[:, 1:2, :, :]
            cr_channel = ycbcr_tensor[:, 2:3, :, :]


            model_input = y_channel / 255.0
            torch.cuda.synchronize()
            start = time()

            x_output = model(model_input)[0] 
            torch.cuda.synchronize()
            end = time()

            rec_y = x_output * 255.0
            rec_y = torch.clamp(rec_y, 0, 255) 


            rec_y_np = rec_y.squeeze().cpu().numpy()
            gt_y_np = y_channel.squeeze().cpu().numpy()

            rec_PSNR = psnr(rec_y_np, gt_y_np)

            rec_SSIM = ssim(rec_y_np, gt_y_np, data_range=255)

            rec_ycbcr = torch.cat([rec_y, cb_channel, cr_channel], dim=1)
            

            rec_rgb = ycbcr_to_rgb(rec_ycbcr)
            rec_rgb = torch.clamp(rec_rgb, 0, 1)
     
            rec_rgb_np = rec_rgb.squeeze().cpu().numpy().transpose(1, 2, 0)
            rec_rgb_np = (rec_rgb_np * 255.0).astype(np.uint8)
            rec_bgr_np = cv2.cvtColor(rec_rgb_np, cv2.COLOR_RGB2BGR)

   
            test_name_split = os.path.split(imgName)
            print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (
                img_no, ImgNum, test_name_split[1], (end - start), rec_PSNR, rec_SSIM))

            resultName = "./%s/%s" % (result_dir, test_name_split[1])
            cv2.imwrite("%s_ratio_%.2f_PSNR_%.2f_SSIM_%.4f.png" % (
                resultName, args.sensing_rate, rec_PSNR, rec_SSIM), rec_bgr_np)

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
            Time_All[0, img_no] = end - start

    print('\n')
    output_data = "CS Reconstruction Result: Ratio %.2f, Avg PSNR %.2f, Avg SSIM %.4f, Avg Time %.4f" % (
        args.sensing_rate, np.mean(PSNR_All), np.mean(SSIM_All), np.mean(Time_All))
    print(output_data)
    print("CS Reconstruction End")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='set14', help='name of test set')
    parser.add_argument('--result_dir', type=str, default='result_krccsnet', help='result directory')
    parser.add_argument('--model', type=str, default='krccsnet', choices=['krccsnet_train','krccsnet'], help='choose model to eval')
    parser.add_argument('--sensing-rate', type=float, default=0.25, help='set sensing rate')

    main()