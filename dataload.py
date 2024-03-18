import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

# 初始化根目录
train_path = 'C:\\Users\\39080\\solarpanel\\train\\'
test_path = 'C:\\Users\\39080\\solarpanel\\test\\'

# 定义读取文件的格式
# 数据集
class MyDataSet(Dataset):
    def __init__(self, data_path: str, transform=None):
        super(MyDataSet, self).__init__()
        self.data_path = data_path
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.transform = transforms

        # 修改 `path_list` 生成逻辑
        self.path_list = []
        for root, dirs, files in os.walk(data_path):
            for subdir in dirs:
                label = int(subdir)
                for filename in os.listdir(os.path.join(root, subdir)):
                    self.path_list.append((os.path.join(root, subdir, filename), label))

    def __getitem__(self, idx: int):
        img_path, label = self.path_list[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

    def __len__(self)->int:
        return len(self.path_list)

train_ds = MyDataSet(train_path)

full_ds = train_ds
train_size = int(0.8*len(full_ds))
test_size = len(full_ds) - train_size
new_train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, test_size])

# 数据加载
new_train_loader = DataLoader(new_train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
new_test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)