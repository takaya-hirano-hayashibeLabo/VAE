import sys
from pathlib import Path
ROOT=Path(__file__).parent.parent
sys.path.append(str(ROOT))

import torch
import torchvision
import torchvision.transforms as transforms
torch.set_default_tensor_type(torch.cuda.FloatTensor)
DEVICE=torch.Tensor([0,0]).device
print(DEVICE)

import matplotlib.pyplot as plt
import random

from src import VAE

def show_images(x, y):
    """
    結果を見る関数
    """
    image_num=5
    fig, axs = plt.subplots(2, image_num, figsize=(2*image_num, 4))

    for i in range(image_num):
        index = random.randint(0, len(x) - 1)
        axs[0, i].imshow(x[index][0], cmap='gray')
        axs[0, i].axis('off')

        axs[1, i].imshow(y[index][0], cmap='gray')
        axs[1, i].axis('off')

    plt.show()


def main():

    #>> データの準備 >>
    class RescaleMinMax:
        """VAEは入力が0から1じゃないといけない"""
        def __call__(self, tensor):
            # 各チャンネルごとの最大値と最小値を計算
            min_val = tensor.min()
            max_val = tensor.max()
            
            # 0から1にスケーリング
            tensor = (tensor - min_val) / (max_val - min_val)
            return tensor

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        RescaleMinMax(),  # 最大値1, 最小値0に正規化
    ])
    trainset = torchvision.datasets.MNIST(root=Path(__file__).parent/'data', 
                                        train=True,
                                        download=True,
                                        transform=transform
                                       )
    trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=100,
                                                shuffle=True,
                                                # num_workers=2,
                                                generator=torch.Generator(device=torch.Tensor([0,0]).device)
                                                )


    testset = torchvision.datasets.MNIST(root=Path(__file__).parent/'data', 
                                            train=False, 
                                            download=True, 
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                                batch_size=100,
                                                shuffle=False, 
                                                # num_workers=2,
                                                generator=torch.Generator(device=torch.Tensor([0,0]).device)
                                                )
    #>> データの準備 >>


    vae:torch.Module=VAE()
    optimizer=torch.optim.Adam(params=vae.parameters(),lr=0.001)
    optimizer.param_groups[0]["caputurable"]=True


    epoches=10
    for epoch in range(epoches):
        vae.train()
        train_loss=0
        for i,data in enumerate(trainloader):
            x,_=data
            y,z,kl_div,loss_rec=vae(x.to(DEVICE))
            loss:torch.Tensor=(loss_rec+kl_div)
            loss.backward()
            train_loss+=loss.item()

            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 99:  # 100ミニバッチごとに進捗を表示
                print(f"loss_reconstraction:{loss_rec}, kl_divergence:{kl_div}")
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss / 100:.3f}')
                train_loss = 0.0


    # テスト
    vae.eval()  # モデルを評価モードに設定
    test_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            x, _ = data
            y,z,kl_div,loss_rec=vae(x.to(DEVICE))
            loss:torch.Tensor=(loss_rec+kl_div)
            test_loss += loss.item()

    test_loss /= len(testloader.dataset)
    print(f'====> Test set loss: {test_loss:.3f}')
    show_images(x.to("cpu").detach().numpy(),y.to("cpu").detach().numpy())


if __name__=="__main__":
    main()

