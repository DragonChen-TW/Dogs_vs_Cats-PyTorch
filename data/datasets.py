import os
#
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms

class DogCatData(data.Dataset):
    def __init__(self, root, mode='train'):
        '''
        Get Image Path.
        '''
        self.mode = mode # 'train', 'test' or 'val'

        if mode == 'train' or mode == 'val':
            folder = os.path.join(root, 'train')
        else:
            folder = os.path.join(root, 'test')
        imgs = [os.path.join(folder, img).replace('\\', '/') \
                for img in os.listdir(folder)]

        # test:  data/test/8973.jpg
        # train: data/train/cat.10004.jpg
        if mode == 'test':
            imgs = sorted(imgs, key=lambda img: int(img.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda img: int(img.split('.')[-2]))

        imgs_num = len(imgs)

        # split into train, val
        if mode == 'test':
            self.imgs = imgs
        elif mode == 'train':
            self.imgs = imgs[:int(imgs_num * 0.8)]
        elif mode == 'val':
            self.imgs = imgs[int(imgs_num * 0.8):]

        # transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                          std=[0.229, 0.224, 0.225])

        # for test and val
        if mode != 'train':
            self.transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        # just for train
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        '''
        return one image's data
        if in test dataset, return image's id
        '''
        img_path = self.imgs[index]
        if self.mode == 'test':
            label = int(img_path.split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0

        data = Image.open(img_path)
        data = self.transforms(data)

        return data, label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    root = 'D:/data/dogs_cats'
    dataset_train = DogCatData(root, mode='train')
    dataset_test  = DogCatData(root, mode='test')
    dataset_val  = DogCatData(root, mode='val')

    train_data = DataLoader(dataset_train,
        shuffle=True, batch_size=16, num_workers=4)
    test_data = DataLoader(dataset_test,
        shuffle=True, batch_size=16, num_workers=4)
    val_data = DataLoader(dataset_val,
        shuffle=True, batch_size=16, num_workers=4)

    print(len(train_data))
    print(len(test_data))
    print(len(val_data))
