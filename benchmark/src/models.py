import argparse

import torch
import torch.nn as nn
import torchvision

# from aves import AVESClassifier  # No longer needed, we use load_feature_extractor instead
from transformers import (
    ClapModel,
    ClapProcessor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)


class ResNetClassifier(nn.Module):
    def __init__(self, model_type, pretrained=False, num_classes=None, multi_label=False):
        super().__init__()

        if model_type.startswith('resnet50'):
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.resnet = torchvision.models.resnet50(weights=weights if pretrained else None)
        elif model_type.startswith('resnet152'):
            weights = torchvision.models.ResNet152_Weights.DEFAULT
            self.resnet = torchvision.models.resnet152(weights=weights if pretrained else None)
        elif model_type.startswith('resnet18'):
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.resnet = torchvision.models.resnet18(weights=weights if pretrained else None)
        else:
            assert False

        self.linear = nn.Linear(in_features=1000, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = x.unsqueeze(1)      # (B, F, L) -> (B, 1, F, L)
        x = x.repeat(1, 3, 1, 1)    # -> (B, 3, F, L)
        x /= x.max()            # normalize to [0, 1]
        # x = self.transform(x)

        x = self.resnet(x)
        logits = self.linear(x)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits


class VGGishClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False):
        super().__init__()

        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.vggish.postprocess = False
        self.vggish.preprocess = False

        self.linear = nn.Linear(in_features=128, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        out = self.vggish(x)
        out = out.reshape(batch_size, -1, out.shape[1])
        outs = out.mean(dim=1)
        logits = self.linear(outs)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits


class BiolingualClassifier(nn.Module):
    def __init__(
        self,
        sample_rate,
        num_classes=None,
        hidden_dim=256,
        dropout=0.1,
        classifier_type="mlp",
        freeze_feature_encoder=False,
        multi_label: bool = False,
    ):
        super().__init__()
        self.processor = ClapProcessor.from_pretrained("davidrrobinson/biolingual")
        self.model = ClapModel.from_pretrained("davidrrobinson/biolingual")
        
        # Freeze/unfreeze CLAP model parameters based on argument
        if freeze_feature_encoder:
            print("Freezing Biolingual feature encoder")
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print("Training Biolingual feature encoder")
            for param in self.model.parameters():
                param.requires_grad = True
        
        input_dim = 512  # CLAP audio feature dimension
        
        if classifier_type == 'mlp':
            # MLP classifier
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        elif classifier_type == 'linear':
            # Linear classifier
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}. Use 'mlp' or 'linear'")
            
        self.loss_func = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
        self.multi_label = multi_label

    def __call__(self, x, y=None):
        device = x.device
        x = x.cpu().numpy()

        processed = self.processor(audios=x, return_tensors="pt", sampling_rate=48000)
        inputs = processed["input_features"].to(device)

        outputs = self.model.get_audio_features(input_features=inputs)
        logits = self.classifier(outputs)

        loss = None
        if y is not None:
            if self.multi_label:
                y = y.float()
            loss = self.loss_func(logits, y)

        return loss, logits


class AvesClassifier(nn.Module):
    def __init__(
        self,
        sample_rate,
        num_classes=None,
        hidden_dim=512,
        dropout=0.1,
        classifier_type="mlp",
        freeze_feature_encoder=False,
        multi_label: bool = False,
    ):
        super().__init__()
        
        # Import the feature extractor from aves
        from aves import load_feature_extractor
        
        if freeze_feature_encoder:
            print("Freezing AVES feature encoder")
        else:
            print("Training AVES feature encoder")

        # Load the AVES feature extractor (without classifier)
        self.feature_extractor = load_feature_extractor(
            config_path="/users/zfne/mustun/Documents/GitHub/aves/aves-bio/aves-base-bio.torchaudio.model_config.json",
            model_path="/users/zfne/mustun/Documents/GitHub/aves/aves-bio/aves-base-bio.torchaudio.pt",
            device="cuda" if torch.cuda.is_available() else "cpu",
            for_inference=False,
        )
        
        embeddings_dim = self.feature_extractor.config.get("encoder_embed_dim", 768)
        
        if freeze_feature_encoder:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        if classifier_type == 'mlp':
            # MLP classifier
            self.classifier = nn.Sequential(
                nn.Linear(embeddings_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        elif classifier_type == 'linear':
            # Linear classifier
            self.classifier = nn.Linear(embeddings_dim, num_classes)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}. Use 'mlp' or 'linear'")
            
        self.loss_func = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
        self.multi_label = multi_label
        self.sample_rate = sample_rate

    def __call__(self, x, y=None):
        # Extract features using AVES feature extractor
        # The feature extractor returns features of shape (batch_size, sequence_length, embedding_dim)
        features = self.feature_extractor.extract_features(x, layers=-1)
        # Average over time dimension to get (batch_size, embedding_dim)
        pooled_features = features.mean(dim=1)
        
        # Apply our classifier
        logits = self.classifier(pooled_features)
        
        loss = None
        if y is not None:
            if self.multi_label:
                y = y.float()
            loss = self.loss_func(logits, y)
            
        return loss, logits


class Dolph2VecClassifier(nn.Module):
    def __init__(
        self,
        sample_rate,
        num_classes=None,
        hidden_dim=512,
        dropout=0.1,
        classifier_type="mlp",
        freeze_feature_encoder=False,
        multi_label: bool = False,
    ):
        super().__init__()
        
        dolph2vec_model = "dolphinteam/model-dolph2vec_type-base_data-DolphinChat_version-v0"
        print(f"Using Dolph2Vec model: {dolph2vec_model}")
        
        # Use the same preprocessor for all variants
        preprocessor_path = "/users/zfne/mustun/Documents/GitHub/Dolph2Vec/dolph2vec-base/preprocessor_config.json"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_json_file(preprocessor_path)
        self.model = Wav2Vec2Model.from_pretrained(dolph2vec_model)

        # Freeze/unfreeze feature encoder based on argument
        if freeze_feature_encoder:
            print("Freezing Dolph2Vec feature encoder")
            self.model.freeze_feature_encoder()
        else:
            print("Training Dolph2Vec feature encoder")

        input_dim = 768  # Dolph2Vec hidden size
        
        if classifier_type == 'mlp':
            # MLP classifier
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        elif classifier_type == 'linear':
            # Linear classifier
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}. Use 'mlp' or 'linear'")
            
        self.loss_func = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
        self.multi_label = multi_label

        self.sample_rate = sample_rate

    def __call__(self, x, y=None):
        device = x.device
        features = self.feature_extractor(
            raw_speech=[w.cpu().numpy() for w in x],
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
            truncation=False,
            # max_length = int(20* self.feature_extractor.sampling_rate),
            # min_length = int(2*self.feature_extractor.sampling_rate)
        )["input_values"]

        features = features.to(device)

        out = self.model(features, output_hidden_states=True)
        pooled = out.hidden_states[-1].mean(1)
        logits = self.classifier(pooled)

        loss = None
        if y is not None:
            if self.multi_label:
                y = y.float()
            loss = self.loss_func(logits, y)

        return loss, logits
