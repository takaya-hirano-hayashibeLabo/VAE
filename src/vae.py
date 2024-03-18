import torch
from torch import nn

from .util import check_in_out_tensor_size


class Encoder(nn.Module):
    def __init__(self):
        """
        CNNで可変サイズにすると実装がめんどいので, 入力と特徴次元は固定する.
        入力は64x64, 出力は32x4x4
        """
        super(Encoder,self).__init__()

        self.conv_net=nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(8,eps=1e-5,),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16,eps=1e-5,),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32,eps=1e-5,),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32,eps=1e-5,),
            nn.AvgPool2d(2),
            nn.ReLU(),
            )
        
        #>> 3DCNNにわたす関係上2次元のままが良いのでflattenかけない >>
        self.mean=nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.log_var=nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        #>> 3DCNNにわたす関係上2次元のままが良いのでflattenかけない >>

        print("Encoder---")
        check_in_out_tensor_size(self.conv_net,tensor_in=torch.rand(size=(128,1,64,64)))

    def forward(self,x):
        
        x=self.conv_net(x)
        mean,log_var=self.mean(x),self.log_var(x)

        return mean,log_var
    

class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder,self).__init__()

        self.conv_net=nn.Sequential(
            nn.ConvTranspose2d(32,32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32,eps=1e-5,),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(16,eps=1e-5,),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(8,eps=1e-5,),
            nn.ReLU(),
            nn.ConvTranspose2d(8,1,kernel_size=4,stride=2,padding=1),
            nn.Sigmoid()
        )

        print("Decoder---")
        check_in_out_tensor_size(self.conv_net,tensor_in=torch.rand(size=(128,32,4,4)))

    def forward(self,x):
        return self.conv_net(x)
    

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()

        self.enc=Encoder()
        self.dec=Decoder()

    def sample_z(self,mean:torch.Tensor,log_var:torch.Tensor):
        """
        平均と分散からサンプリングする関数
        """
        epsilon=torch.randn(size=mean.shape)
        return mean+epsilon*torch.exp(0.5*log_var) #μ+N(0,σ)

    def forward(self,x):
        """
        :return y
        :return z
        :return kl_div
        :return loss_rec
        """
        mean,log_var=self.enc(x)
        z=self.sample_z(mean,log_var)
        y=self.dec(z)
        kl_div=-0.5 * torch.mean(1 + log_var - mean**2 - torch.exp(log_var))
        loss_rec=-torch.mean(x * torch.log(y + 1e-15) + (1 - x) * torch.log(1 - y + 1e-15))

        return y,z,kl_div,loss_rec
