import SimpleITK
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

from typing import (
    Dict,
)
from abc import abstractmethod
import torch

from evalutils.evalutils import Algorithm


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def to_input_format(image):
    val_transforms = A.Compose(
    [
        A.Resize(width=420, height=420),
        ToTensorV2(),
    ]
    )
    image = val_transforms(image = image)["image"]
    input_image = image.unsqueeze(0).to(device)
    #change to float
    return input_image

def unpack_single_output(output):
    #return output.cpu().numpy().astype(float)[0]
    return output.cpu().numpy().astype(float)


class MultiClassAlgorithm(Algorithm):

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Classify input_image image
        return self.predict(input_image=input_image)

    @abstractmethod
    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        raise NotImplementedError()

    def save(self):
        if len(self._case_results) > 1:
            raise RuntimeError("Multiple case prediction not supported with single-value output interfaces.")
        case_result = self._case_results[0]

        for output_file, result in case_result.items():
            with open(str(self._output_path / output_file) + '.json', "w") as f:
                json.dump(result, f)


if __name__ == "__main__":
    MultiClassAlgorithm()
