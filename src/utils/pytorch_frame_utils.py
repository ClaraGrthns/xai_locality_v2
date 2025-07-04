import torch
from torch_frame import stype, TaskType
from torch_frame.data.tensor_frame import TensorFrame
import torch_frame
import numpy as  np


def transform_logit_to_class_proba(logit):
    """
    Transforms logits to class probabilities.

    This function takes a tensor of logits and converts them to class probabilities
    using the appropriate activation function. If the logits have a single class
    (binary classification), the sigmoid function is used. For multi-class classification,
    the softmax function is applied.

    Args:
        logit (torch.Tensor): A tensor of logits with shape (N, C) where N is the number
                              of samples and C is the number of classes.
                              if C = 1, the sigmoid function is applied.
                              if C > 1, the softmax function is applied.
    Returns:
        torch.Tensor: A tensor of class probabilities with the same shape as the input logits.
    """
    if logit.shape[-1] == 1:
        predictions_sm = torch.sigmoid(logit)
        predictions_sm = torch.cat([1 - predictions_sm, predictions_sm], dim=-1)
    else:
        predictions_sm = torch.softmax(logit, dim=-1)
    return predictions_sm

def load_dataframes(data_path):
    """
    Load dataframes from a specified path and convert them to tensors.

    Args:
        data_path (str): The path to the file containing the dataframes.

    Returns:
        tuple: A tuple containing three tensors:
            - trn_feat: Tensor containing the training features.
            - val_feat: Tensor containing the validation features.
            - tst_feat: Tensor containing the test features.
    """
    data = torch.load(data_path, map_location=torch.device('cpu'), weights_only=False)
    train_tensor_frame, val_tensor_frame, test_tensor_frame = data["train"], data["val"], data["test"]
    trn_feat = tensorframe_to_tensor(train_tensor_frame)
    val_feat = tensorframe_to_tensor(val_tensor_frame)
    tst_feat = tensorframe_to_tensor(test_tensor_frame)
    return trn_feat, val_feat, tst_feat

def tensorframe_to_tensor(tf):
    """
    Converts a tensor frame to a single concatenated tensor.

    Args:
        tf: A tensor frame object that contains a dictionary of feature tensors.

    Returns:
        A single tensor obtained by concatenating the categorical, numerical, 
        and embedding feature tensors along the second dimension.

    The tensor frame object `tf` is expected to have a `feat_dict` attribute 
    which is a dictionary containing the following keys:
        - stype.categorical: Tensor of categorical features.
        - stype.numerical: Tensor of numerical features.
        - stype.embedding: Tensor of embedding features.

    The function checks for the presence of these keys in the `feat_dict` and 
    concatenates the corresponding tensors if they exist.
    """
    tst_feat = tf.feat_dict
    dfs = []
    if stype.categorical in tst_feat:
        feat_tensor = tst_feat[stype.categorical]
        dfs.append(feat_tensor)

    if stype.numerical in tst_feat:
        feat_tensor = tst_feat[stype.numerical]
        dfs.append(feat_tensor)

    if stype.embedding in tst_feat:
        feat = tst_feat[stype.embedding]
        feat = feat.values
        feat = feat.view(feat.size(0), -1)
        dfs.append(feat)
    return torch.cat(dfs, dim=1)

 
def tensor_to_tensorframe(tensor, col_names_dict):
    return TensorFrame({torch_frame.numerical: tensor}, col_names_dict) 


class PytorchFrameWrapper(torch.nn.Module):
    def __init__(self, original_model, col_names_dict):
        super(PytorchFrameWrapper, self).__init__()
        self.original_model = original_model
        self.col_names_dict = col_names_dict

    def forward(self, input_tensor):
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        tensor_frame = self._convert_to_tensor_frame(input_tensor)
        return self.original_model(tensor_frame)

    def _convert_to_tensor_frame(self, input_tensor):
        return tensor_to_tensorframe(input_tensor, self.col_names_dict)