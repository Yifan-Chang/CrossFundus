import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from flair.pretraining.data.transforms import augmentations_pretraining
from .transformer_decoder import TQN_Model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import numpy as np

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    #print(image.shape[0], stepSize[1])
    for y in range(0, image.shape[0], stepSize[1]):
        for x in range(0, image.shape[1], stepSize[0]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# 返回滑动窗结果集合，本示例暂时未用到
def get_slice(image, stepSize, windowSize):
    slice_sets = []
    for (x, y, window) in sliding_window(image, stepSize, windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
            continue
        slice = image[y:y + windowSize[1], x:x + windowSize[0]]
        slice = slice.permute(3,2,0,1)
        slice_sets.append(slice)
    return slice_sets

def get_local_patch(image, stepSize, windowSize):

    image = image.permute(2,3,1,0)

    slice_sets = get_slice(image,stepSize, windowSize)

    return slice_sets



class AdapterWrapper(object):
    def __init__(self, model, targets, tta=False, fta=False):
        '''
        targets：类别  category     tta：测试阶段数据增强  test time data augmentation   fta：训练时数据增强  train/fit time data augmentation
        '''
        self.model = copy.deepcopy(model)
        self.model.eval()                               # 冻结编码器参数  Freezing encoder parameter
        self.num_targets = len(targets)                 # 类别个数  Number of classes
        self.tta = tta
        self.fta = fta
        self.number_augmentations = 20                  # 训练增强次数  enhancement times for fta/tta

    # 获取视觉特征和标签  Get visual features and labels
    def extract_vision_features(self, data_loader, transforms=None):
        self.model.eval()
        epoch_iterator = tqdm(data_loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True)

        # 对于后续的适配器 输入是CLIP视觉编码器得到的特征向量 输出是类别号
        # adapter input is the feature vector of CLIP visual encoder and the output is the class number
        X, Y = [], []
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)

            with torch.no_grad():
                if transforms is not None:
                    images = transforms(images)

                #print(images.shape)
                #images = F.interpolate(images, (224,224))#自然权重
                x = self.model.vision_model(images)
                #x = self.model.encode_image(images)
                #print(x.shape)

            X.extend(x.cpu().detach().numpy())
            Y.extend(batch["label"].numpy())

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    # 训练的接口：进行数据增强、抽视觉特征、调用训练函数
    # Training interface:  data augmentation, extract visual features, call training functions
    def fit(self, loaders, transforms=None):
        data_loader = loaders["train"]                                                         # 训练集  img path 标签/mask

        # 是否使用训练增强策略 增加训练数据  use augmentation strategies to increase training data?
        if self.fta:
            transforms = augmentations_pretraining
        # 获取视觉特征  get visual features
        if self.fta and transforms is not None:
            X, Y = [], []
            for i in range(self.number_augmentations):
                Xa, Ya = self.extract_vision_features(data_loader, transforms=transforms)
                X.append(Xa), Y.append(Ya)
            X = np.concatenate(X, 0)                                                       # 合并成一维  Merge into one dimension
            Y = np.concatenate(Y, 0)
        else:
            X, Y = self.extract_vision_features(data_loader, transforms=transforms)

        self.train(X, Y)

    # 训练用虚函数  Virtual functions for training
    def train(self, X, Y):
        """
        虚函数 由具体适配器实现  Implemented by a specific adapter
        """
        return

    # 预测用虚函数  Virtual functions for prediction
    def predict(self, loader, transforms=None):
        """
        虚函数 由具体适配器实现  Implemented by a specific adapter
        """
        return
"""
多模态适配器   Multimodal adapter
"""
# 多模态适配器父类      继承适配器父类；增加文本特征的抽取
# Multimodal adapter parent class inherits the adapter parent class; Increase the extraction of text features
class LanguageAdapterWrapper(AdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, tta=tta, fta=fta)

        # 输入类别名称    输出对应类别的文本特征（有/无领域知识）
        # Input category name. Output text characteristics corresponding to the category (with/without domain knowledge)
        self.text_embeds_dict, self.text_embeds = model.compute_text_embeddings(list(targets.keys()), domain_knowledge=domain_knowledge)


class adapter_module_cf(nn.Module):
    def __init__(self, cache_keys, cache_values, text_feat):
        super().__init__()
      
        self.tip_layer = torch.nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False)
        self.tip_layer.weight = torch.nn.Parameter(cache_keys.t())  
        self.cache_values = cache_values
        self.beta = 5
        self.cross_attn = TQN_Model()
        self.local_crs_attn = TQN_Model()
        self.text_feat = text_feat
        self.text_adapter = torch.nn.Linear(512, 512, bias=False)
        self.alpha_l = torch.nn.Parameter(torch.ones(1))
        self.omega = torch.nn.Parameter(torch.tensor(0.5))
        self.linear_i2t = nn.Linear(512 * 2, 512)
        self.gate_i2t = nn.Linear(512 * 2, 512)
        self.dropout = nn.Dropout(0.8)
  

    def forward(self, X_t, X_p):

        tip = self.tip_layer(X_t)

        crs_attn = self.cross_attn(X_t.unsqueeze(1), self.text_feat).squeeze(2)
        loc_crs_attn = self.local_crs_attn(X_p, self.text_feat).squeeze(2)

        c = nn.Softmax(1)(loc_crs_attn) @ self.text_feat #b,4
        input_cat = torch.cat([X_t, c], 1)
        input_1 = F.tanh(self.linear_i2t(input_cat))
        input_1 = self.dropout(input_1)
        gate = torch.sigmoid(self.gate_i2t(input_cat))
        output = X_t * gate + input_1 * (1 - gate)

        c_loc = nn.Softmax(1)(crs_attn) @ self.text_feat #b,4
        b, p, f = X_p.shape
        c_loc = c_loc.unsqueeze(1).repeat(1, X_p.shape[1], 1)
        input_cat = torch.cat([X_p, c_loc], 2).reshape(-1, 1024)
        input_1 = F.tanh(self.linear_i2t(input_cat))
        input_1 = self.dropout(input_1)
        gate = torch.sigmoid(self.gate_i2t(input_cat))
        output_loc = X_p.reshape(-1,512) * gate + input_1 * (1 - gate)
        output_loc = output_loc.reshape(b, p, f)


        crs_attn_1 = self.cross_attn(output.unsqueeze(1), self.text_feat).squeeze(2)
        loc_crs_attn_1 = self.local_crs_attn(output_loc, self.text_feat).squeeze(2)     
        
        crs_attn = (0.25*crs_attn_1 + 0.75*crs_attn)
        loc_crs_attn = (0.25*loc_crs_attn_1 + 0.75*loc_crs_attn)

        zs = 100 * X_t @ self.text_feat.t()

        crs_attn = self.omega * zs  + 0.8*(crs_attn) + 0.2*loc_crs_attn

        return  tip, crs_attn

class CrossFundus(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False, train=False):
        '''
        train: 是否训练
        '''
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.beta = 5
        self.alpha = 1
        self.gamma  = 0.005 #AMD:0.05, FIVE:0.75
        self.cache_keys = []            # 存视觉特征向量  visual feature vectors
        self.cache_values = []          # 存标签 one-hot形式  label one-hot vector
        self.targets = targets

        # 是否训练适配器  train?
        self.train_tip = train
        self.adapter_layer = []         # 存放适配器  adapter
        #self.text_embeds_dict, self.text_embeds, self.global_text_emb = model.compute_text_embeddings_gl(list(targets.keys()), domain_knowledge=domain_knowledge) 

    def predict(self, loader, transforms=None):
        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = self.adapter(X)               # X: N * embed 测试集  test set data
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
            epoch_iterator = tqdm(loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True)
# adapter input is the feature vector of CLIP visual encoder and the output is the class number
            X, Y = [], []
            X_P=  []
            for step, batch in enumerate(epoch_iterator):
                images = batch["image"].to(device).to(torch.float32)

                with torch.no_grad():
                    if transforms is not None:
                        images = transforms(images)

                    
                    im_list = get_local_patch(images,[images.shape[3]//2, images.shape[2]//2], [images.shape[3]//2, images.shape[2]//2])
                    im_list_tensor = torch.stack(im_list).transpose(0,1)
                    b, p = im_list_tensor.shape[0], im_list_tensor.shape[1]
              
                    im_list_tensor = im_list_tensor.reshape(im_list_tensor.shape[0]*im_list_tensor.shape[1], im_list_tensor.shape[2],im_list_tensor.shape[3],im_list_tensor.shape[4])
                    im_list_tensor = F.interpolate(im_list_tensor, scale_factor=2, mode='bicubic')
                    x = self.model.vision_model(images)
                    x_p = self.model.vision_model(im_list_tensor)
                    x_p = x_p.reshape(b,p,512)
                    

                X.extend(x.cpu().detach().numpy())
                X_P.extend(x_p.cpu().detach().numpy()) 
           

            X = np.array(X)
            X_P = np.array(X_P)
   
            
            X = torch.tensor(X).to(device)
            X_P = torch.tensor(X_P).to(device)
            with torch.no_grad():
                score = self.adapter(X,X_P)

        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()
        return refs, preds
    
    def fit(self, loaders, transforms=None):
        data_loader = loaders["train"]                                                         # 训练集  img path 标签/mask
    
        self.train(data_loader, transforms=transforms)


    def train(self, data_loader, transforms):

        self.model.eval()
        epoch_iterator = tqdm(data_loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True)

        #self.text_embeds_dict, self.text_embeds = self.model.compute_text_embeddings(list(self.targets.keys()), domain_knowledge=False)
        # 对于后续的适配器 输入是CLIP视觉编码器得到的特征向量 输出是类别号
        # adapter input is the feature vector of CLIP visual encoder and the output is the class number
        X, Y = [], []
        X_P = []
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)

            with torch.no_grad():
                if transforms is not None:
                    images = transforms(images)

                x = self.model.vision_model(images)
                #print(images.shape)
                im_list = get_local_patch(images,[images.shape[3]//2, images.shape[2]//2], [images.shape[3]//2, images.shape[2]//2])
                im_list_tensor = torch.stack(im_list).transpose(0,1)
                b, p = im_list_tensor.shape[0], im_list_tensor.shape[1]
                #print(b,p)
                im_list_tensor = im_list_tensor.reshape(im_list_tensor.shape[0]*im_list_tensor.shape[1], im_list_tensor.shape[2],im_list_tensor.shape[3],im_list_tensor.shape[4])
                im_list_tensor = F.interpolate(im_list_tensor, scale_factor=2, mode='bicubic')
                x = self.model.vision_model(images)
                x_p = self.model.vision_model(im_list_tensor)
                x_p = x_p.reshape(b,p,512)
     


            X.extend(x.cpu().detach().numpy()) 
            X_P.extend(x_p.cpu().detach().numpy()) 
            Y.extend(batch["label"].numpy())

        X = np.array(X)
        X_P = np.array(X_P)
        Y = np.array(Y)

        X = torch.tensor(X)
        X_P = torch.tensor(X_P)
        Y = torch.tensor(Y)

        self.cache_keys = torch.transpose(X.to(device), 1, 0).to(torch.float32).to(device)              # embed * N'  自己和自己算相似度（训练集）  count similarity with itself (training set)
        self.cache_values = torch.nn.functional.one_hot(Y).to(torch.float32).to(device) 

        self.cross_adapter = adapter_module_cf(self.cache_keys, self.cache_values, self.text_embeds).to(device)

        epochs, lr, bs = 60, 0.001, 1
        optimizer = torch.optim.AdamW(self.cross_adapter.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * X.shape[0])
        indexes = np.arange(0, X.shape[0])
        random.shuffle(indexes)

        for i_epoch in range(epochs):
            loss_epoch = 0.0
            for i_sample in range(X.shape[0]):
                X_batch = X[indexes[i_sample], :].unsqueeze(0).to(device)                           # 1 * embd
                target = Y[indexes[i_sample]].unsqueeze(0).to(device)
                X_P_batch = X_P[indexes[i_sample], :].unsqueeze(0).to(device)                               # 1，
                
                      
                affinity, crs = self.cross_adapter(X_batch, X_P_batch)

                clip_logits = crs
                                                             # 1 * embed ==》 1 * N'  （以训练集为参照 初始化权重）  Initialize the weights with reference to the training set
                cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values   # 1 * K （以训练集为参照）  Use the training set as a reference
                cache_logits /= X.shape[0]
                cache_logits *= self.model.logit_scale.exp()
    
                
                tip_logits = clip_logits + cache_logits * self.cross_adapter.alpha_l
                loss = torch.nn.functional.cross_entropy(tip_logits, target) 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_epoch += loss.item()/X.shape[0]
    # 适配器 + 相似度计算  Adapter + similarity calculation
    def adapter(self, X, X_P):
        # X：N * embed   测试集的视觉特征  Visual feature of the test set
        clip_logits = 100 * (X @ self.text_embeds.t().to(device))           
        affinity, crs_ = self.cross_adapter(X, X_P)
        cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values      
        clip_logits = crs_
        logits = clip_logits + cache_logits * self.cross_adapter.alpha_l

        return logits

