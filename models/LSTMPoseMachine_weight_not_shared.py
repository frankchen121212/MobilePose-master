import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class LPM_raw(nn.Module):

    def __init__(self,in_channels, hidden_channels, out_channels, device, T=5):
        super(LPM_raw,self).__init__()
        self.k = 14
        #ConvNet1
        self.conv1_stage1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7_stage1 = nn.Conv2d(512, self.k + 1, kernel_size=1,padding=0)
        self.max_pool_stage1 = nn.MaxPool2d(3, stride=2)

        #ConvNet2 #Frame1
        self.conv1_stage2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.max_pool_stage2 = nn.MaxPool2d(3, stride=2)
        #LSTM #Frame1
        lstm_size = hidden_channels + out_channels + 1
        self.g_x_stage2 = nn.Conv2d(in_channels=lstm_size,out_channels=lstm_size,kernel_size=3,padding=1,bias=True)
        self.i_x_stage2 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.o_x_stage2 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)

        #ConvNet3 #Frame1
        self.Mconv1_stage2 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage2 = nn.Conv2d(128,self.k + 1 , 1 , padding=0)

        # ConvNet2 #Frame2
        self.conv1_stage3 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2_stage3 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage3 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv4_stage3 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.max_pool_stage3 = nn.MaxPool2d(3, stride=2)

        # LSTM # Frame2
        self.g_x_stage3 = nn.Conv2d(in_channels=lstm_size,out_channels=lstm_size,kernel_size=3,padding=1,bias=True)
        self.g_h_stage3 = nn.Conv2d(in_channels=lstm_size,out_channels=lstm_size,kernel_size=3,padding=1,bias=True)
        self.g_stage3 = nn.BatchNorm2d(num_features=lstm_size,momentum=0.1)
        self.f_x_stage3 = nn.Conv2d(in_channels=lstm_size,out_channels=lstm_size,kernel_size=3,padding=1,bias=True)
        self.f_h_stage3 = nn.Conv2d(in_channels=lstm_size,out_channels=lstm_size,kernel_size=3,padding=1,bias=True)
        self.f_stage3 = nn.BatchNorm2d(num_features=lstm_size,momentum=0.1)
        self.i_x_stage3 = nn.Conv2d(in_channels=lstm_size,out_channels=lstm_size,kernel_size=3,padding=1,bias=True)
        self.i_h_stage3 = nn.Conv2d(in_channels=lstm_size,out_channels=lstm_size,kernel_size=3,padding=1,bias=True)
        self.i_stage3 = nn.BatchNorm2d(num_features=lstm_size,momentum=0.1)
        self.o_x_stage3 = nn.Conv2d(in_channels=lstm_size,out_channels=lstm_size,kernel_size=3,padding=1,bias=True)
        self.o_h_stage3 = nn.Conv2d(in_channels=lstm_size,out_channels=lstm_size,kernel_size=3,padding=1,bias=True)
        self.o_stage3 = nn.BatchNorm2d(num_features=lstm_size,momentum=0.1)

        #ConvNet3 #Frame2
        self.Mres1_stage3 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mres2_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mres3_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mres4_stage3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mres5_stage3 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        # ConvNet2 #Frame3
        self.conv1_stage4 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2_stage4 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage4 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv4_stage4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.max_pool_stage4 = nn.MaxPool2d(3, stride=2)

        # LSTM # Frame3
        self.g_x_stage4 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.g_h_stage4= nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.g_stage4 = nn.BatchNorm2d(num_features=lstm_size, momentum=0.1)
        self.f_x_stage4 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.f_h_stage4 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.f_stage4 = nn.BatchNorm2d(num_features=lstm_size, momentum=0.1)
        self.i_x_stage4 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.i_h_stage4 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.i_stage4 = nn.BatchNorm2d(num_features=lstm_size, momentum=0.1)
        self.o_x_stage4 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.o_h_stage4 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.o_stage4 = nn.BatchNorm2d(num_features=lstm_size, momentum=0.1)

        # ConvNet3 #Frame3
        self.Mres1_stage4 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mres2_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mres3_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mres4_stage4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mres5_stage4 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        # ConvNet2 #Frame4
        self.conv1_stage5 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2_stage5 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage5 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv4_stage5 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.max_pool_stage5 = nn.MaxPool2d(3, stride=2)

        # LSTM # Frame4
        self.g_x_stage5 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.g_h_stage5 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.g_stage5 = nn.BatchNorm2d(num_features=lstm_size, momentum=0.1)
        self.f_x_stage5 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.f_h_stage5 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.f_stage5 = nn.BatchNorm2d(num_features=lstm_size, momentum=0.1)
        self.i_x_stage5 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.i_h_stage5 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.i_stage5 = nn.BatchNorm2d(num_features=lstm_size, momentum=0.1)
        self.o_x_stage5 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.o_h_stage5 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.o_stage5 = nn.BatchNorm2d(num_features=lstm_size, momentum=0.1)

        # ConvNet3 #Frame4
        self.Mres1_stage5 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mres2_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mres3_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mres4_stage5 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mres5_stage5 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        # ConvNet2 #Frame5
        self.conv1_stage6 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2_stage6 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage6 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv4_stage6 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.max_pool_stage6 = nn.MaxPool2d(3, stride=2)

        # LSTM # Frame5
        self.g_x_stage6 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.g_h_stage6 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.g_stage6 = nn.BatchNorm2d(num_features=lstm_size, momentum=0.1)
        self.f_x_stage6 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.f_h_stage6 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.f_stage6 = nn.BatchNorm2d(num_features=lstm_size, momentum=0.1)
        self.i_x_stage6 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.i_h_stage6 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.i_stage6 = nn.BatchNorm2d(num_features=lstm_size, momentum=0.1)
        self.o_x_stage6 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.o_h_stage6 = nn.Conv2d(in_channels=lstm_size, out_channels=lstm_size, kernel_size=3, padding=1, bias=True)
        self.o_stage6 = nn.BatchNorm2d(num_features=lstm_size, momentum=0.1)

        # ConvNet3 #Frame5
        self.Mres1_stage6 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mres2_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mres3_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mres4_stage6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mres5_stage6 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

    def forward(self, x, centers):
        x = torch.split(x, 1, dim=1)
        centers = F.avg_pool2d(centers, 9, stride=8)
        x_0 = torch.squeeze(x[0], 1)
        x_1 = torch.squeeze(x[1], 1)
        x_2 = torch.squeeze(x[2], 1)
        x_3 = torch.squeeze(x[3], 1)
        x_4 = torch.squeeze(x[4], 1)

        # ConvNet1
        b_0 = F.relu(self.conv1_stage1(x_0))
        b_0 = self.max_pool_stage1(b_0)
        b_0 = F.relu(self.conv2_stage1(b_0))
        b_0 = self.max_pool_stage1(b_0)
        b_0 = F.relu(self.conv3_stage1(b_0))
        b_0 = self.max_pool_stage1(b_0)
        b_0 = F.relu(self.conv4_stage1(b_0))
        b_0 = F.relu(self.conv5_stage1(b_0))
        b_0 = F.relu(self.conv6_stage1(b_0))
        b_0 = F.relu(self.conv7_stage1(b_0))

        beliefs = []
        beliefs_loss=[]
        beliefs_loss.append(b_0)
        # ConvNet2 #Frame1 x=b_0
        b_prev, h_prev, c_prev = b_0, None, None
        f_1 = F.relu(self.conv1_stage2(x_0))
        f_1 = self.max_pool_stage2(f_1)
        f_1 = F.relu(self.conv2_stage2(f_1))
        f_1 = self.max_pool_stage2(f_1)
        f_1 = F.relu(self.conv3_stage2(f_1))
        f_1 = self.max_pool_stage2(f_1)
        f_1 = F.relu(self.conv4_stage2(f_1))


        f_1 = torch.cat([f_1, b_prev, centers], dim=1)

        #LSTM #Frame1
        i_2 = torch.sigmoid(self.i_x_stage2(f_1))
        o_2 = torch.sigmoid(self.o_x_stage2(f_1))
        g_1 = torch.sigmoid(self.g_x_stage2(f_1))
        c_1 = i_2.mul(g_1)
        h_1 = o_2.mul(torch.tanh(c_1))

        # ConvNet3 #Frame1
        b_1 = F.relu(self.Mconv1_stage2(h_1))
        b_1 = F.relu(self.Mconv2_stage2(b_1))
        b_1 = F.relu(self.Mconv3_stage2(b_1))
        b_1 = F.relu(self.Mconv4_stage2(b_1))
        b_1 = F.relu(self.Mconv5_stage2(b_1))
        beliefs.append(b_1)
        beliefs_loss.append(b_1)
        # ConvNet2 #Frame2  x=x_1
        b_prev, h_prev, c_prev = b_1, h_1, c_1

        f_2 = F.relu(self.conv1_stage3(x_1))
        f_2 = self.max_pool_stage3(f_2)
        f_2 = F.relu(self.conv2_stage3(f_2))
        f_2 = self.max_pool_stage3(f_2)
        f_2 = F.relu(self.conv3_stage3(f_2))
        f_2 = self.max_pool_stage3(f_2)
        f_2 = F.relu(self.conv4_stage3(f_2))
        f_2 = torch.cat([f_2, b_prev, centers], dim=1)

        # LSTM #Frame2
        i_2 = torch.sigmoid(self.i_x_stage3(f_2) + self.i_h_stage3(h_prev))
        i_2 = self.i_stage3(i_2)
        o_2 = torch.sigmoid(self.o_x_stage3(f_2) + self.o_h_stage3(h_prev))
        o_2 = self.o_stage3(o_2)
        g_2 = torch.sigmoid(self.g_x_stage3(f_2) + self.g_h_stage3(h_prev))
        g_2 = self.g_stage3(g_2)
        f = torch.sigmoid(self.f_x_stage3(f_2) + self.f_h_stage3(h_prev))
        f = self.f_stage3(f)
        c_2 = i_2.mul(g_2).add(f.mul(c_prev))
        h_2 = o_2.mul(torch.tanh(c_2))

        # ConvNet3 #Frame2
        b_2 = F.relu(self.Mres1_stage3(h_2))
        b_2 = F.relu(self.Mres2_stage3(b_2))
        b_2 = F.relu(self.Mres3_stage3(b_2))
        b_2 = F.relu(self.Mres4_stage3(b_2))
        b_2 = F.relu(self.Mres5_stage3(b_2))
        beliefs.append(b_2)
        beliefs_loss.append(b_2)
        # ConvNet2 #Frame3  x=x_2
        b_prev, h_prev, c_prev = b_2, h_2, c_2
        f_3 = F.relu(self.conv1_stage4(x_2))
        f_3 = self.max_pool_stage4(f_3)
        f_3 = F.relu(self.conv2_stage4(f_3))
        f_3 = self.max_pool_stage4(f_3)
        f_3 = F.relu(self.conv3_stage4(f_3))
        f_3 = self.max_pool_stage4(f_3)
        f_3 = F.relu(self.conv4_stage4(f_3))
        f_3 = torch.cat([f_3, b_prev, centers], dim=1)

        # LSTM #Frame3
        i_3 = torch.sigmoid(self.i_x_stage4(f_3) + self.i_h_stage4(h_prev))
        i_3 = self.i_stage4(i_3)
        o_3 = torch.sigmoid(self.o_x_stage4(f_3) + self.o_h_stage4(h_prev))
        o_3 = self.o_stage4(o_3)
        g_3 = torch.sigmoid(self.g_x_stage4(f_3) + self.g_h_stage4(h_prev))
        g_3 = self.g_stage4(g_3)
        f = torch.sigmoid(self.f_x_stage4(f_3) + self.f_h_stage4(h_prev))
        f = self.f_stage4(f)
        c_3 = i_3.mul(g_3).add(f.mul(c_prev))
        h_3 = o_3.mul(torch.tanh(c_3))

        # ConvNet3 #Frame3
        b_3 = F.relu(self.Mres1_stage4(h_3))
        b_3 = F.relu(self.Mres2_stage4(b_3))
        b_3 = F.relu(self.Mres3_stage4(b_3))
        b_3 = F.relu(self.Mres4_stage4(b_3))
        b_3 = F.relu(self.Mres5_stage4(b_3))
        beliefs.append(b_3)
        beliefs_loss.append(b_3)
        # ConvNet2 #Frame4  x=x_3
        b_prev, h_prev, c_prev = b_3, h_3, c_3
        f_4 = F.relu(self.conv1_stage5(x_3))
        f_4 = self.max_pool_stage5(f_4)
        f_4 = F.relu(self.conv2_stage5(f_4))
        f_4 = self.max_pool_stage5(f_4)
        f_4 = F.relu(self.conv3_stage5(f_4))
        f_4 = self.max_pool_stage5(f_4)
        f_4 = F.relu(self.conv4_stage5(f_4))
        f_4 = torch.cat([f_4, b_prev, centers], dim=1)

        # LSTM #Frame4
        i_4 = torch.sigmoid(self.i_x_stage5(f_4) + self.i_h_stage5(h_prev))
        i_4 = self.i_stage5(i_4)
        o_4 = torch.sigmoid(self.o_x_stage5(f_4) + self.o_h_stage5(h_prev))
        o_4 = self.o_stage5(o_4)
        g_4 = torch.sigmoid(self.g_x_stage5(f_4) + self.g_h_stage5(h_prev))
        g_4 = self.g_stage5(g_4)
        f = torch.sigmoid(self.f_x_stage5(f_4) + self.f_h_stage5(h_prev))
        f = self.f_stage5(f)
        c_4 = i_4.mul(g_4).add(f.mul(c_prev))
        h_4 = o_4.mul(torch.tanh(c_4))

        # ConvNet3 #Frame4
        b_4 = F.relu(self.Mres1_stage5(h_4))
        b_4 = F.relu(self.Mres2_stage5(b_4))
        b_4 = F.relu(self.Mres3_stage5(b_4))
        b_4 = F.relu(self.Mres4_stage5(b_4))
        b_4 = F.relu(self.Mres5_stage5(b_4))
        beliefs.append(b_4)
        beliefs_loss.append(b_4)
        # ConvNet2 #Frame5  x=x_4
        b_prev, h_prev, c_prev = b_4, h_4, c_4
        f_5 = F.relu(self.conv1_stage6(x_4))
        f_5 = self.max_pool_stage5(f_5)
        f_5 = F.relu(self.conv2_stage6(f_5))
        f_5 = self.max_pool_stage5(f_5)
        f_5 = F.relu(self.conv3_stage6(f_5))
        f_5 = self.max_pool_stage5(f_5)
        f_5 = F.relu(self.conv4_stage6(f_5))
        f_5 = torch.cat([f_5, b_prev, centers], dim=1)

        # LSTM #Frame5
        i_5 = torch.sigmoid(self.i_x_stage6(f_5) + self.i_h_stage6(h_prev))
        i_5 = self.i_stage6(i_5)
        o_5 = torch.sigmoid(self.o_x_stage6(f_5) + self.o_h_stage6(h_prev))
        o_5 = self.o_stage6(o_5)
        g_5 = torch.sigmoid(self.g_x_stage6(f_5) + self.g_h_stage6(h_prev))
        g_5 = self.g_stage6(g_5)
        f = torch.sigmoid(self.f_x_stage6(f_5) + self.f_h_stage6(h_prev))
        f = self.f_stage6(f)
        c_5 = i_5.mul(g_5).add(f.mul(c_prev))
        h_5 = o_5.mul(torch.tanh(c_5))

        # ConvNet3 #Frame5
        b_5 = F.relu(self.Mres1_stage6(h_5))
        b_5 = F.relu(self.Mres2_stage6(b_5))
        b_5 = F.relu(self.Mres3_stage6(b_5))
        b_5 = F.relu(self.Mres4_stage6(b_5))
        b_5 = F.relu(self.Mres5_stage6(b_5))
        beliefs.append(b_5)
        beliefs_loss.append(b_5)
        out = torch.stack(beliefs, 1)
        out_loss = torch.stack(beliefs_loss, 1)
        return out ,out_loss



    def init_weights(self,model_dir,model,dataset,pretrained):
        path = os.path.join(model_dir,model,dataset,pretrained)

        if os.path.isfile(path):
            path = os.path.join(model_dir, model, dataset, pretrained)

            if os.path.isfile(path):
                path = os.path.join(model_dir, model, dataset, pretrained)
                pretrained_dict = torch.load(path)
                print(pretrained_dict.keys())
                for my_key in self.state_dict().keys():
                    for load_key in pretrained_dict.keys():
                        if my_key == load_key:
                            print('key matched--{}'.format(my_key))
                            print('{}'.format(self.state_dict()[my_key].shape))
                            print('{}\n'.format(pretrained_dict[load_key].shape))
                print('=>Loading LPM wheight not shared model from --{}'.format(path))
                self.load_state_dict(pretrained_dict, strict=False)

            else:
                print('=> CPM pretrained model dose not exist')
                print('=> please download it first')
                raise ValueError('CPM pretrained model does not exist')
