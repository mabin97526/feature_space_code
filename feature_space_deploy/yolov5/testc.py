import os
import torch
from PIL import Image

'./data/customdata\\segimages/val'
'./data/customdata\\seglabels/val'
'./data/customdata\\seglabels/val\\segmentation_10.png'

if __name__ == '__main__':
    t1 = torch.Tensor([[1,2,3],[4,5,6]])
    t2 = torch.Tensor([[11, 12, 13], [14, 15, 16]])
    t3 = torch.Tensor([[21, 22],
                                   [23, 24]])
    r = torch.concat([t1,t2],-2)
    print(r)