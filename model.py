import torch
import torch.nn as nn
import torch.nn.functional as F


class CPM(nn.Module):

    def __init__(self, k ):
        super(CPM, self).__init__()

        self.k = k # 16

        ### Stage 1 ###

        self.stage1 = nn.Sequential(
            # 9*9
            nn.Conv2d(3, 128, 9, padding=4),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 9*9
            nn.Conv2d(128, 128, 9, padding=4),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 9*9
            nn.Conv2d(128, 128, 9, padding=4),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 5*5
            nn.Conv2d(128, 32, 5, padding=2),
            nn.ReLU(True),
            # 9*9
            nn.Conv2d(32, 512, 9, padding=4),
            nn.ReLU(True),
            # 1*1
            nn.Conv2d(512, 512, 1),
            nn.ReLU(True),
            # 1*1
            nn.Conv2d(512, self.k+1, 1)
        )

        ### Stage 2 ###
        ### Not same with CPM paper ###
        self.middle = nn.Sequential(
            # 9*9
            nn.Conv2d(3, 128, 9, padding=4),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 9*9
            nn.Conv2d(128, 128, 9, padding=4),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 9*9
            nn.Conv2d(128, 128, 9, padding=4),
            nn.ReLU(True),  
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 5*5
        self.conv_stage2 = nn.Conv2d(128, 32, 5, padding=2)

        self.stage2 = nn.Sequential(

            # concat + 11*11
            nn.Conv2d(self.k+2+32, 128, 11, padding=5),
            nn.ReLU(True),
            # 11*11
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(True),
            # 11*11
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(True),
            # 1*1
            nn.Conv2d(128, 128,1,padding=0),
            nn.ReLU(True),
            # 1*1
            nn.Conv2d(128, self.k+1, 1, padding=0),
        )

        ### Stage 3 ###

         # 5*5
        self.conv_stage3 = nn.Conv2d(128, 32, 5, padding=2)

        self.stage3 = nn.Sequential(
            # concat + 11*11
            nn.Conv2d(self.k +2+ 32, 128, 11,padding=5),
            nn.ReLU(True),
            # 11*11
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(True),
            # 11*11
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(True),
            # 1*1
            nn.Conv2d(128, 128,1,padding=0),
            nn.ReLU(True),
            # 1*1
            nn.Conv2d(128, self.k+1, 1, padding=0),
        )

        ### Stage 4 ###
        
        # 5*5
        self.conv_stage4 = nn.Conv2d(128, 32, 5, padding=2)

        self.stage4 = nn.Sequential(
        
            # concat + 11*11
            nn.Conv2d(self.k +2+ 32, 128, 11,padding=5),
            nn.ReLU(True),
            # 11*11
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(True),
            # 11*11
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(True),
            # 1*1
            nn.Conv2d(128, 128,1,padding=0),
            nn.ReLU(True),
            # 1*1
            nn.Conv2d(128, self.k+1, 1, padding=0),
        )

        ### Stage 5 ###
        
        # 5*5
        self.conv_stage5 = nn.Conv2d(128, 32, 5, padding=2)

        self.stage5 = nn.Sequential(
        
            # concat + 11*11
            nn.Conv2d(self.k +2+ 32, 128, 11,padding=5),
            nn.ReLU(True),
            # 11*11
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(True),
            # 11*11
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(True),
            # 1*1
            nn.Conv2d(128, 128,1,padding=0),
            nn.ReLU(True),
            # 1*1
            nn.Conv2d(128, self.k+1, 1, padding=0),
        )

        ### Stage 6 ###
        
        # 5*5
        self.conv_stage6 = nn.Conv2d(128, 32, 5, padding=2)

        self.stage6 = nn.Sequential(
        
            # concat + 11*11
            nn.Conv2d(self.k +2+ 32, 128, 11,padding=5),
            nn.ReLU(True),
            # 11*11
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(True),
            # 11*11
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(True),
            # 1*1
            nn.Conv2d(128, 128,1,padding=0),
            nn.ReLU(True),
            # 1*1
            nn.Conv2d(128, self.k+1, 1, padding=0),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)


    def stage_1(self, x):
        return self.stage1(x)

    def stage_2(self, x, score1, pool_center_map):
        x = F.relu(self.conv_stage2(x))
        #print(x.shape, score1.shape, pool_center_map.shape)
        out = torch.cat([x, score1, pool_center_map], dim=1)
        return self.stage2(out)

    def stage_3(self, x, score2, pool_center_map):
        x = F.relu(self.conv_stage3(x))
        out = torch.cat([x, score2, pool_center_map], dim=1)
        return self.stage3(out)
    
    def stage_4(self,x, score3, pool_center_map):
        x = F.relu(self.conv_stage4(x))
        out = torch.cat([x, score3, pool_center_map], dim=1)
        return self.stage4(out)
    
    def stage_5(self,x, score4, pool_center_map):
        x = F.relu(self.conv_stage5(x))
        out = torch.cat([x, score4, pool_center_map], dim=1)
        return self.stage4(out)
    
    def stage_6(self,x, score5, pool_center_map):
        x = F.relu(self.conv_stage6(x))
        out = torch.cat([x, score5, pool_center_map], dim=1)
        return self.stage4(out)
    
    def forward(self, x, centermap):
        pool_center_map = self.avg_pool(centermap)

        feature = self.middle(x)
        score1 = self.stage_1(x)            
        score2 = self.stage_2(feature, score1, pool_center_map)
        score3 = self.stage_3(feature, score2, pool_center_map)
        score4 = self.stage_4(feature, score3, pool_center_map)
        score5 = self.stage_5(feature, score4, pool_center_map)
        score6 = self.stage_6(feature, score5, pool_center_map)

        return score1, score2, score3, score4, score5, score6

