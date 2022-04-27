from torch import nn, optim
from typing import Dict
from pathlib import Path
import SimpleITK
import torch
import torchvision
import tqdm
#import efficientnet_pytorch

from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from utils import MultiClassAlgorithm, to_input_format, unpack_single_output, device
from algorithm.preprocess import preprocess


COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn1 = torchvision.models.vgg19()
        self.cnn2 = torchvision.models.vgg19()
        self.cnn3 = torchvision.models.vgg19()
        
        self.fc1 = nn.Linear(32+32+32, 32)
        self.fc2 = nn.Linear(32, 1)
        
        for model in [self.cnn1, self.cnn2, self.cnn3]:
            model.classifier[6] = nn.Sequential(nn.Linear(model.classifier[6].in_features, 32))
                                                
    
    
    def forward(self, image1, image2, image3, gender=None, age=None):
        x1 = self.cnn1(image1)
        x2 = self.cnn2(image2)
        x3 = self.cnn3(image3)
        
        x = torch.cat((x1,x2,x3), dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StoicAlgorithm(MultiClassAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/input/images/ct/"),
            output_path=Path("/output/")
        )

        # load model
        #load_model1 = torch.load('./algorithm/model_cvd.pth', map_location=torch.device(device))
        #load_model2 = torch.load('./algorithm/model_svr.pth')
        
        #self.model1 = load_model1['model']
        #self.model1 = self.model1.to(device)
        #self.model1.load_state_dict(load_model1['state_dict'])
        #self.model1 = self.model1.eval()
        
        #self.model2 = load_model2['model']
        #self.model2 = self.model2.to(device)
        #self.model2.load_state_dict(load_model2['state_dict'])
        #self.model2 = self.model2.eval()

    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        # pre-processing
        input_image = preprocess(input_image)
        input_image1 = input_image[0]
        input_image2 = input_image[1]
        input_image3 = input_image[2]

        # run model
        with torch.no_grad():
            #output1 = torch.sigmoid(self.model1(input_image1, input_image2, input_image3))
            output1 = torch.sigmoid(model1(input_image1, input_image2, input_image3))
            output2 = torch.sigmoid(model2(input_image1, input_image2, input_image3))
        prob_covid = unpack_single_output(output1)
        prob_severe = unpack_single_output(output2)
        #print(prob_covid[0][0])
        #print(prob_severe[0][0])

        return {
            COVID_OUTPUT_NAME: prob_covid[0][0],
            SEVERE_OUTPUT_NAME: prob_severe[0][0]
        }


if __name__ == "__main__":
    load_model1 = torch.load('./algorithm/model_cvd.pth', map_location=torch.device(device))
    load_model2 = torch.load('./algorithm/model_svr.pth', map_location=torch.device(device))
        
    model1 = load_model1['model']
    model1 = model1.to(device)
    model1.load_state_dict(load_model1['state_dict'])
    model1 = model1.eval()
    
    model2 = load_model2['model']
    model2 = model2.to(device)
    model2.load_state_dict(load_model2['state_dict'])
    model2 = model2.eval()
        
    StoicAlgorithm().process()
