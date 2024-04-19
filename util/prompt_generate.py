from segment_anything import SamPredictor, sam_model_registry

import random
import torch
import numpy as np

class PromptGenerator():
    def __init__(self,img,pred,gt,
                 device = 'cuda',
                 sam_mode = "vit_h",
                 checkpoint=r"/raid/shiqi/sam_pretrained/sam_vit_h_4b8939.pth"):
        #############################
        # img : (3, h, w) ndarray
        # pred : (1,2,256,256) tensor
        # gt : (1,256,256) tensor
        ###########################
        self.pred = pred.argmax(dim=1)[0]
        self.gt = gt
        self.device = device
        self.original_size = self.pred.shape[-2:]
        sam = sam_model_registry[sam_mode](checkpoint=checkpoint)
        sam.to(device=device)
        self.mask_predictor = SamPredictor(sam)
        self.mask_predictor.set_image(img)

    def rand_one_choice(self,coordinates):
        return random.choice(coordinates)

    def rand_multi_choice(self, coordinates, n):
        selected_coordinates = random.sample(coordinates, n)
        return selected_coordinates

    def generate_points(self,num,label=None):
        # label [0,0,1]
        # assert len(label) == num
        _, H, W = self.gt.size()
        err = self.gt - self.pred # (1, 256, 256)
        positive_coordinates = [[i, j] for i in range(H) for j in range(W) if err[0, i, j] > 0]
        negative_coordinates = [[i, j] for i in range(H) for j in range(W) if err[0, i, j] < 0]
        print(len(positive_coordinates))
        print('=======================')
        print(len(negative_coordinates))
        self.point_coords = []
        if label == None:
            self.point_labels = []
            if len(positive_coordinates) >= len(negative_coordinates) and len(positive_coordinates) >= num:
                self.point_labels = [1] * num
                self.point_coords = self.rand_multi_choice(positive_coordinates,num)
            elif len(positive_coordinates) < len(negative_coordinates) and len(negative_coordinates) >= num:
                self.point_labels = [0] * num
                self.point_coords = self.rand_multi_choice(negative_coordinates, num)
            elif len(positive_coordinates) >= len(negative_coordinates) and len(positive_coordinates) < num:
                self.point_labels = [1] * len(positive_coordinates)
                self.point_coords = positive_coordinates
                self.point_labels += [0]*(num-len(positive_coordinates))
                self.point_coords += self.rand_multi_choice(negative_coordinates,num-len(positive_coordinates))
            else:
                self.point_labels = [0] * len(negative_coordinates)
                self.point_coords = negative_coordinates
                self.point_labels += [1] * (num - len(negative_coordinates))
                self.point_coords += self.rand_multi_choice(positive_coordinates, num - len(negative_coordinates))

        else:
            self.point_labels = label.sort()
            for _l in self.point_labels:
                if _l == 0:
                    self.point_coords.append(self.rand_one_choice(negative_coordinates))
                else:
                    self.point_coords.append(self.rand_one_choice(positive_coordinates))
        return np.array(self.point_coords), np.array(self.point_labels)

    def generate_box(self):
        # 找到值大于 0 的像素的坐标
        y_indices, x_indices = torch.where(self.gt[0] > 0)

        # 计算包围框的坐标
        x_min, x_max = x_indices.min().item(), x_indices.max().item()
        y_min, y_max = y_indices.min().item(), y_indices.max().item()

        # 添加扰动到包围框坐标
        _, H, W = self.gt.shape
        x_min = max(0, x_min - torch.randint(0, 20, (1,)).item())
        x_max = min(W, x_max + torch.randint(0, 20, (1,)).item())
        y_min = max(0, y_min - torch.randint(0, 20, (1,)).item())
        y_max = min(H, y_max + torch.randint(0, 20, (1,)).item())

        # 创建包围框坐标的列表 [x_min, y_min, x_max, y_max]
        bbox = [x_min, y_min, x_max, y_max]

        return np.array(bbox)

    def get_prompt_embedding(self,num_points,point_label=None,box=False):
        if box:
            bbox = self.generate_box()
            box = self.mask_predictor.transform.apply_boxes(bbox, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
            sparse_emb,_ = self.mask_predictor.model.prompt_encoder(points=None,boxes=box_torch,masks=None)
            # sparse_emb (1,2,256)
            return sparse_emb[0], [1,1]

        if num_points > 0:
            point_coords, point_labels = self.generate_points(num_points, point_label)
            point_coords = self.mask_predictor.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            sparse_emb,_ = self.mask_predictor.model.prompt_encoder(points=(coords_torch, labels_torch),boxes=None,masks=None)
            return sparse_emb[0,:-1,:], point_labels












##################################
# TEST
#################################

# 创建大小为（1，2，256，256）的随机 tensor A，数值在 0 到 1 之间
tensor_A = torch.rand(1, 2, 256, 256)

# 创建大小为（1，256，256）的全零 tensor B，并将其数据类型转换为 long
tensor_B = torch.zeros(1, 256, 256).long()

# 定义圆的参数
center_x, center_y = 128, 128  # 圆心坐标
radius = 50  # 圆的半径

# 生成圆的掩码
y_indices, x_indices = torch.meshgrid(torch.arange(256), torch.arange(256))
circle_mask = ((x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2).long()

# 在 tensor B 中使用圆的掩码将对应位置设置为 1
tensor_B[0][circle_mask == 1] = 1

prompt = PromptGenerator(img=np.random.randint(0, 256, size=(3,256,256), dtype=np.uint8),
                         pred=tensor_A,
                         gt=tensor_B)
_c,_l = prompt.get_prompt_embedding(num_points=6)
print(_c.shape,_l.shape)

# prompt = PromptGenerator(tensor_A, tensor_B)
# p_c,p_l = prompt.generate_points(5)
# bbox = prompt.generate_box()
# print(p_c, p_l)
# print(bbox)

# device = 'cuda'
# original_size = (256,256)
# sam = sam_model_registry["vit_h"](checkpoint=r"/raid/shiqi/sam_pretrained/sam_vit_h_4b8939.pth")
# sam.to(device=device)
# mask_predictor = SamPredictor(sam)
# mask_predictor.set_image(np.random.randint(0, 256, size=(3,256,256), dtype=np.uint8))
# p_c = mask_predictor.transform.apply_coords(p_c, original_size)
# coords_torch = torch.as_tensor(p_c, dtype=torch.float, device=device)
# labels_torch = torch.as_tensor(p_l, dtype=torch.int, device=device)
# coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
#
# box = mask_predictor.transform.apply_boxes(bbox, original_size)
# box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
# box_torch = box_torch[None, :]
# point_emb,_ = mask_predictor.model.prompt_encoder(points=(coords_torch, labels_torch),boxes=box_torch,masks=None)
# point_emb,_ = mask_predictor.model.prompt_encoder(points=None,boxes=box_torch,masks=None)
# print(point_emb.shape) #torch.Size([1, 3, 256])

